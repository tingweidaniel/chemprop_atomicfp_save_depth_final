from argparse import Namespace
from typing import List, Union

import torch
import torch.nn as nn
import numpy as np

from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function

torch.set_printoptions(edgeitems=7)


class MPNEncoder(nn.Module):  # for atomic_vecs_d2, atomic_vecs_final, mol_vecs
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args
        self.AM_dim0 = args.AM_dim0
        self.AM_dim1 = args.AM_dim1
        
        
        if self.features_only:
            return
        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size
                
        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)
            
        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)
    
    # zero-padding tensor
    def padding(self, mol_vector, padding_final_size=42):
        if self.AM_dim0:
            padding_final_size=43
        num_atoms_index = mol_vector.shape[0]
        num_features_index = mol_vector.shape[1]
        padding_tensor = torch.zeros((padding_final_size, num_features_index))
        padding_tensor[:num_atoms_index, :] = mol_vector
        return padding_tensor

    def forward(self,
                mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            if self.args.cuda:
                features_batch = features_batch.cuda()

            if self.features_only:
                return features_batch

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()

        if self.atom_messages:
            a2a = mol_graph.get_a2a()

        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

            if self.atom_messages:
                a2a = a2a.cuda()

        # Input
        if self.atom_messages:
            #print('atom_messages')
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
            
        message = self.act_func(input)  # num_bonds x hidden_size


        save_depth = []  # wei, update for each step


        # wei, save the information of depth=0 (atomic features)
        padding = torch.zeros((f_atoms.size()[0], f_atoms.size()[1]+self.hidden_size))
        if self.args.cuda:
            padding = padding.cuda()
        padding[:, :f_atoms.size()[1]] = f_atoms
        a_input = padding
        #print('mol_graph:', mol_graph.smiles_batch)
        #print('a_input', a_input)
        #print('a_input.size()', a_input.size())
        atom_hiddens = self.act_func(self.W_o(a_input))
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
        save_depth.append(atom_hiddens)


        # Message passing
        for depth in range(self.depth - 1):

            ################ save information of one bond distance from the central atom ################
            if depth == 0:
                a2x = a2a if self.atom_messages else a2b
                nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
                atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
                atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
                save_depth.append(atom_hiddens)
            ################ save information of one bond distance from the central atom ################

            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden
       
            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden
            
            # wei, save depth
            a2x = a2a if self.atom_messages else a2b
            nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
            a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
            a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
            atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
            atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
            save_depth.append(atom_hiddens)  # save information of each depth

        ############ origin ############
        #print('message_after_m_passing:\n', message)
        #a2x = a2a if self.atom_messages else a2b
        #nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        #a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        #a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        #atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        #atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
        ############ origin ############    
    
        # Readout
        atomic_vecs_d0 = []
        atomic_vecs_d1 = []
        atomic_vecs_d2 = []
        atomic_vecs_final = []
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                #print('save_depth[0].size()', save_depth[0].size())
                #print('save_depth[1].size()', save_depth[1].size())
                #print('save_depth[-1].size()', save_depth[2].size())
                #print('len save_depth', len(save_depth))

                cur_hiddens_d0 = save_depth[0].narrow(0, a_start, a_size)
                cur_hiddens_d1 = save_depth[1].narrow(0, a_start, a_size)
                cur_hiddens_d2 = save_depth[2].narrow(0, a_start, a_size)
                cur_hiddens_final = save_depth[-1].narrow(0, a_start, a_size)
                cur_hiddens_mol = save_depth[-1].narrow(0, a_start, a_size)

                atomic_vec_d0 = cur_hiddens_d0
                atomic_vec_d0 = self.padding(atomic_vec_d0)  # padding
                atomic_vec_d1 = cur_hiddens_d1
                atomic_vec_d1 = self.padding(atomic_vec_d1)  # padding
                atomic_vec_d2 = cur_hiddens_d2
                atomic_vec_d2 = self.padding(atomic_vec_d2)  # padding
                atomic_vec_final = cur_hiddens_final
                atomic_vec_final = self.padding(atomic_vec_final)  # padding
                
                mol_vec = cur_hiddens_mol  # (num_atoms, hidden_size)
                mol_vec = mol_vec.sum(dim=0) / a_size

                atomic_vecs_d0.append(atomic_vec_d0)
                atomic_vecs_d1.append(atomic_vec_d1)
                atomic_vecs_d2.append(atomic_vec_d2)
                atomic_vecs_final.append(atomic_vec_final)
                mol_vecs.append(mol_vec)

        atomic_vecs_d0 = torch.stack(atomic_vecs_d0, dim=0)
        atomic_vecs_d1 = torch.stack(atomic_vecs_d1, dim=0)
        atomic_vecs_d2 = torch.stack(atomic_vecs_d2, dim=0)
        atomic_vecs_final = torch.stack(atomic_vecs_final, dim=0)  
        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        if self.args.cuda:
            atomic_vecs_d0, atomic_vecs_d1, atomic_vecs_d2, atomic_vecs_final, mol_vecs = atomic_vecs_d0.cuda(), atomic_vecs_d1.cuda(), atomic_vecs_d2.cuda(), atomic_vecs_final.cuda(), mol_vecs.cuda()

        #overall_vecs=torch.cat((mol_vecs,mol_vec_molar),dim=0)
        if self.use_input_features:
            features_batch = features_batch.to(mol_vecs)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1,features_batch.shape[0]])
            mol_vecs = torch.cat([mol_vecs, features_batch], dim=1)  # (num_molecules, num_atoms,  hidden_size)

        return atomic_vecs_d0, atomic_vecs_d1, atomic_vecs_d2, atomic_vecs_final, mol_vecs
    
 

class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, num_atoms, hidden_size) containing the encoding of each molecule.
        """
        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch, self.args)

        output_d0, output_d1, output_d2, output_final, output_mol = self.encoder.forward(batch, features_batch)
        #print('MPN_atomic len', len(self.encoder.forward(batch, features_batch)))
        
        return output_d0, output_d1, output_d2, output_final, output_mol

