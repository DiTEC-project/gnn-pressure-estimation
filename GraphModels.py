import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MLP, GINConv, ARMAConv, ChebConv, GCNConv, GCN2Conv
from torch_geometric.nn import BatchNorm

from typing import List, Optional, Union
from torch import Tensor
from torch.nn import ( Dropout, Sequential, SELU)
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.nn.conv import SimpleConv

class GATConvNet(torch.nn.Module):
    def __init__(self, net_params):
        super(GATConvNet, self).__init__()
        # torch.manual_seed(42)
        self.net_params = net_params

        self.convs = torch.nn.ModuleList()

        in_channels = net_params["input_dim"]
        for _ in range(net_params["num_layers"] - 1):
            self.convs.append(GATConv(in_channels,
                                      net_params['hidden_dim'],
                                      heads=net_params['heads'], concat=True))  # aggr=net_params["aggr"]
            in_channels = net_params["heads"] * net_params["hidden_dim"]

        self.convs.append(GATConv(net_params["heads"] * net_params["hidden_dim"],
                                  net_params["out_dim"], heads=1, concat=False))  # aggr=net_params["aggr"]

        self.skips = torch.nn.ModuleList()
        self.skips.append(Linear(net_params["input_dim"], net_params["heads"] * net_params["hidden_dim"]))
        for _ in range(net_params["num_layers"] - 2):
            self.skips.append(
                Linear(net_params["heads"] * net_params["hidden_dim"], net_params["heads"] * net_params["hidden_dim"]))
        self.skips.append(Linear(net_params["heads"] * net_params["hidden_dim"], net_params["out_dim"]))

    def forward(self, x, edge_index, batch):
        for i in range(self.net_params["num_layers"] - 1):
            x = F.relu(self.convs[i](x, edge_index) + self.skips[i](x))
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.convs[-1](x, edge_index) + self.skips[-1](x)
        return torch.sigmoid(x)


class GATConvNoSkipsNet(torch.nn.Module):
    def __init__(self, net_params):
        super(GATConvNoSkipsNet, self).__init__()
        self.conv1 = GATConv(1, 64, 2, concat=True)
        self.conv2 = GATConv(128, 64, 2, concat=True)
        self.conv3 = GATConv(128, 1, 1, concat=False)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)


class ARMAConvNet(torch.nn.Module):
    def __init__(self, net_params):
        super(ARMAConvNet, self).__init__()
        self.net_params = net_params
        self.convs = torch.nn.ModuleList()

        in_channels = net_params["input_dim"]
        for _ in range(net_params["num_layers"] - 1):
            self.convs.append(ARMAConv(in_channels, net_params["hidden_dim"], aggr=net_params["aggr"]))
            in_channels = net_params["hidden_dim"]

        self.convs.append(ARMAConv(net_params["hidden_dim"], net_params["out_dim"]))

    def forward(self, x, edge_index, batch):
        for i in range(self.net_params["num_layers"] - 1):
            x = F.relu(self.convs[i](x, edge_index))

        x = self.convs[self.net_params["num_layers"] - 1](x, edge_index)

        return torch.sigmoid(x)

class GCNConvNet(torch.nn.Module):
    def __init__(self, net_params):
        super(GCNConvNet, self).__init__()

        # torch.manual_seed(42)
        self.net_params = net_params

        self.convs = torch.nn.ModuleList()

        input_dim = net_params["input_dim"]
        for _ in range(net_params["num_layers"] - 1):
            self.convs.append(GCNConv(in_channels=input_dim,
                                      out_channels=net_params["hidden_dim"], cached=False,
                                      aggr=net_params["aggr"]))
            input_dim = net_params["hidden_dim"]

        self.convs.append(GCNConv(net_params["hidden_dim"], net_params["hidden_dim"], cached=False,
                                  aggr=net_params["aggr"]))
        self.lin1 = torch.nn.Linear(net_params["hidden_dim"], int(net_params["hidden_dim"] / 2))
        self.lin2 = torch.nn.Linear(int(net_params["hidden_dim"] / 2), net_params["out_dim"])

    def forward(self, x, edge_index, batch):
        for i in range(self.net_params["num_layers"] - 1):
            x = F.relu(self.convs[i](x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return torch.sigmoid(x)


class GCNBaseNet(torch.nn.Module):
    def __init__(self, net_params):
        super(GCNBaseNet, self).__init__()
        self.net_params = net_params

        self.conv1 = GCNConv(net_params["input_dim"], net_params["hidden_dim"], aggr=net_params["aggr"])
        self.conv2 = GCNConv(net_params["hidden_dim"], net_params["out_dim"], aggr=net_params["aggr"])

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)


class MixerMLP(torch.nn.Module):
    def __init__(self,name, net_params):
        super(MixerMLP, self).__init__()
        self.name = 'mixer_mlp'
        # torch.manual_seed(42)
        self.net_params = net_params
        self.linear1 = torch.nn.Linear(5855, 1024)
        self.linear2 = torch.nn.Linear(1024, 5855)

    def forward(self, x, edge_index, batch):
        ori = x
        x = x.squeeze(-1)
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = x.view_as(ori)
        return x
#######################################################################
#ref: https://github.com/BME-SmartLab/GraphConvWat/blob/be97b45fbc7dfdba22bb1ee406424a7c568120e5/model/richmond.py
class GraphConvWat(torch.nn.Module):
    def __init__(self,name, in_channels, out_channels):
        super().__init__()
        self.name=name
        self.block1 = ChebConv(in_channels, 120, K=240)
        self.block2 = ChebConv(120, 60, K=120)
        self.block3 = ChebConv(60, 30, K=20)
        self.block4 = ChebConv(30, out_channels, K=1, bias=False)

    def forward(self, x, edge_index, batch, edge_attr):
        x = F.silu(self.block1(x, edge_index))
        x = F.silu(self.block2(x, edge_index))
        x = F.silu(self.block3(x, edge_index))
        x = self.block4(x, edge_index)
        return x

class ChebNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, name='ChebNet', nc = 32):
        super(ChebNet, self).__init__()
        self.name=name 
        self.block1 = ChebConv(in_channels, nc, K=24)
        self.block2 = ChebConv(nc, nc, K=12)
        self.block3 = ChebConv(nc, nc, K=10)
        self.block4 = ChebConv(nc, out_channels, K=1, bias=False)

    def forward(self, x, edge_index, batch, edge_attr):
        x = F.silu(self.block1(x, edge_index))
        x = F.silu(self.block2(x, edge_index))
        x = F.silu(self.block3(x, edge_index))
        x = self.block4(x, edge_index)
        return x

#####################################################################################################

class GCN2(torch.nn.Module):
    def __init__(self,name='GCN2',num_blocks = 64, nc = 32, in_channels=1, out_channels=1):
        super(GCN2, self).__init__()
        self.num_blocks = num_blocks
        self.name= f'{name}_{num_blocks}b_{nc}c'
        blocks= []

        for i in range(self.num_blocks):
            layer = GCN2Conv(nc,alpha=0.1,theta=0.5,layer=i+1)
            blocks.append(layer)
        self.blocks = torch.nn.ModuleList(blocks)
        self.steam = Linear(in_channels=in_channels, out_channels=nc)
        self.lin = Linear(in_channels=nc, out_channels=out_channels)
    
    def forward(self, x, edge_index, batch=None, edge_attr=None):
        x = self.steam(x)
        x_0 = x
        for i in range(self.num_blocks):
            x = self.blocks[i](x=x, x_0=x_0 , edge_index=edge_index) 
        x = self.lin(x)
        return x
    
class GAT(torch.nn.Module):
    def __init__(self,name='GAT',num_blocks = 10, nc = 32, in_channels=1, out_channels=1):
        super(GAT, self).__init__()
        self.num_blocks = num_blocks
        self.name= f'{name}_{num_blocks}b_{nc}c'
        blocks= []

        for i in range(self.num_blocks):
            if i == 0 :
                layer = GATConv(in_channels,nc,heads=2,concat=True)
            elif i == self.num_blocks - 1:
                layer = GATConv(2*nc,out_channels,heads=1,concat=True)
            else:
                layer = GATConv(2*nc,nc,heads=2,concat=True)
            blocks.append(layer)
        self.blocks = torch.nn.ModuleList(blocks)
    
    def forward(self, x, edge_index, batch=None, edge_attr=None):
        for i in range(self.num_blocks):
            x = self.blocks[i](x, edge_index) 
        return x


class GIN(torch.nn.Module):
    def __init__(self,name='GIN_bottleneck',num_blocks = 10, nc = 32, in_channels=1, out_channels=1):
        super(GIN, self).__init__()
        self.name = f'{name}_{num_blocks}b_{nc}c'
        
        self.num_blocks = num_blocks

        blocks= []

        for i in range(self.num_blocks):
            if i == 0 :
                layer = GINConv(MLP(dims=[in_channels,nc//2,nc]),eps=0.0)
            elif i == self.num_blocks - 1:
                layer = GINConv(Linear(nc,out_channels,bias=False),eps=0.0)
            else:
                layer = GINConv(MLP(dims=[nc,nc//2,nc]),eps=0.0)
            blocks.append(layer)
        self.blocks = torch.nn.ModuleList(blocks)
       

    
    def forward(self, x, edge_index, batch=None, edge_attr=None):
        for i in range(self.num_blocks):
            o=x
            x = self.blocks[i](x, edge_index)
            if x.shape[-1] == o.shape[-1]:
              x = x + o
        return x
    

#####################################################################################################
#REF: https://github.com/HammerLabML/GCNs_for_WDS
class MLP(Sequential):
    def __init__(self, dims: List[int], bias: bool = True, dropout: float = 0., activ=SELU()):
        m = []
        for i in range(1, len(dims)):
            m.append(Linear(dims[i - 1], dims[i], bias=bias))

            if i < len(dims) - 1:                
                m.append(activ)
                m.append(Dropout(dropout))

        super().__init__(*m)

class GENConvolution(MessagePassing):
    r"""
    Args:
        in_dim (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_dim (int): Size of each output sample.
        edge_dim (int): Size of edge features.
        aggr (str, optional): The aggregation scheme to use (:obj:`"softmax"`,
            :obj:`"softmax_sg"`, :obj:`"power"`, :obj:`"add"`, :obj:`"mean"`,
            :obj:`max`). (default: :obj:`"softmax"`)        
        num_layers (int, optional): The number of MLP layers.
            (default: :obj:`2`)
        eps (float, optional): The epsilon value of the message construction
            function. (default: :obj:`1e-7`)
        bias (bool, optional): If set to :obj:`False`, will not use bias. 
            (default: :obj:`True`)
        dropout (float, optional): Percentage of neurons to be dropped in MLP.
            (default: :obj:`0.`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GenMessagePassing`.
    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{in}), (|\mathcal{V_t}|, F_{t})`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge attributes :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int,
                 aggr: str = 'add', num_layers: int = 2, eps: float = 1e-7, 
                 bias: bool = True, dropout: float = 0., **kwargs):

        kwargs.setdefault('aggr', None)
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.aggr = aggr
        self.eps = eps
        
        assert aggr in ['add', 'mean', 'max']

        dims = [self.in_dim]
        for i in range(num_layers - 1):
            dims.append(2 * in_dim)
        dims.append(self.out_dim)
        self.mlp = MLP(dims, bias=bias, dropout=dropout)

        """ Added a linear layer to manage dimensionality """
        #print(f'res in = {in_dim+edge_dim}')
        #print(f'res out = {in_dim}')
        self.res = Linear(in_dim + edge_dim, in_dim, bias=bias)

    def reset_parameters(self):
        if self.msg_norm is not None:
            self.msg_norm.reset_parameters()
        if self.t and isinstance(self.t, Tensor):
            self.t.data.fill_(self.initial_t)
        if self.p and isinstance(self.p, Tensor):
            self.p.data.fill_(self.initial_p)

    #def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
    #            edge_attr: OptTensor = None, size: Size = None, 
    #            residual: bool = True, mlp: bool = True) -> Tensor:
    def forward(self, x, edge_index,
                batch=None,
                edge_attr=None, size= None, 
                residual = True, mlp = True) -> Tensor:
        """"""
        #print(f'index = {gcn_index}')           
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)        
        x_in = x[0]
        
        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        if edge_attr is not None:
          sndr_node_attr = torch.gather(x_in, 0, edge_index[0:1,:].repeat(x_in.shape[1], 1).T)
          rcvr_node_attr = torch.gather(x_in, 0, edge_index[1:2,:].repeat(x_in.shape[1], 1).T)
          
          edge_attr = edge_attr + (sndr_node_attr - rcvr_node_attr).abs()
          
        latent = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=size) 
        
        """ Added a linear layer to manage dimensionality """
        if mlp:
            latent = self.res(latent)
        else:
            latent = torch.tanh(self.res(latent))

        #del sndr_node_attr, rcvr_node_attr
        
        if residual:
            latent = latent + x[1]
        
        #del x, edge_index, edge_attr 
        if mlp:
            latent = self.mlp(latent)  
        return latent       

    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        """ Concatenating edge features instead of adding those to node features """
        
        msg = x_j if edge_attr is None else torch.cat((x_j, edge_attr), dim=1)
        #print(f'msg.shape = {msg.shape}')
        #del x_j, edge_attr
        return F.selu(msg) + self.eps

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:

        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_dim}, '
                f'{self.out_dim}, aggr={self.aggr})')
        
class m_GCN(torch.nn.Module):
    def __init__(self,name=None, in_dim=1, out_dim=1, edge_dim=0, latent_dim=32, n_aggr=45, n_hops=1, bias=False, num_layers=2, dropout=0., batch_size=32, w_sigmoid=True):
        super(m_GCN, self).__init__()
        self.name = f'mGCN-n_aggr{n_aggr}-nhops{n_hops}-nmlp{num_layers}' if name is None else name
        self.n_aggr = n_aggr
        self.n_hops = n_hops
        self.batch_size = batch_size
        self.latent = latent_dim
        self.out_dim = out_dim
        self.w_sigmoid = w_sigmoid
        self.node_in = torch.nn.Linear(in_dim, latent_dim, bias=bias)        
        self.node_out = torch.nn.Linear(latent_dim, out_dim, bias=bias)
        self.edge = torch.nn.Linear(edge_dim, latent_dim, bias=bias)        
            
        self.gcn_aggrs = torch.nn.ModuleList()
        for _ in range(n_aggr):
            gcn = GENConvolution(latent_dim, latent_dim, latent_dim, aggr="add", bias=bias, num_layers=num_layers, dropout=dropout)
            self.gcn_aggrs.append(gcn)

        
    def forward(self, x, edge_index, batch = None, edge_attr=None):
        #print(f'edge_attr type = {type(edge_attr)}')
        
        #print(f'edge_index shape = {edge_index.shape}')
        #print(f'sndr_node_attr shape = {sndr_node_attr.shape}')
        #print(f'rcvr_node_attr shape = {rcvr_node_attr.shape}')
        #print(f'x_in shape = {x_in.shape}')
        #print(f'edge_attr shape = {edge_attr.shape}')
        """ Embedding for edge features. """
        if edge_attr is not None:
            edge_attr = self.edge(edge_attr)
        """ Embedding for node features. """
        Z = self.node_in(x)

        """ 
            Mutiple GCN layers.
        """
        for i,gcn in enumerate(self.gcn_aggrs):
            """
                Multiple Hops.
            """
            #print(f'checking gcn_{i}')
            for _ in range(self.n_hops - 1):
                Z = torch.selu(gcn(x=Z, edge_index=edge_index, edge_attr=edge_attr,  mlp=False))
            Z = torch.selu(gcn(x=Z, edge_index=edge_index, edge_attr=edge_attr,  mlp=True))
                
        """ Reconstructing node features through a final dense layer. """
        y_predict = self.node_out(Z)
        if self.w_sigmoid:
            y_predict = F.sigmoid(y_predict)
        return y_predict
    
###############################################################################################


class GResBlockMeanConv(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hc):
        super(GResBlockMeanConv, self).__init__()

        self.conv1 = GATConv(in_dim, hc, 2, concat=True)
        self.conv2 = GATConv(hc * 2, out_dim, 1, concat=False)
        self.mean_conv = SimpleConv(aggr="mean")

    def forward(self, x, edge_index, edge_attr=None):
        x_0 = torch.clone(x)
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = self.mean_conv(x, edge_index) + x_0
        x = F.relu(x)
        return x


class GATResMeanConv(torch.nn.Module):
    def __init__(self, name='GATResMeanConv', num_blocks=5, nc = 32):
        super(GATResMeanConv, self).__init__()

        self.num_blocks = num_blocks

        self.lin0 = Linear(1, nc)
        self.blocks = torch.nn.ModuleList()
        self.name = name
        for _ in range(self.num_blocks):
            block = GResBlockMeanConv(nc, nc, nc)
            self.blocks.append(block)

        self.lin1 = Linear(nc, 1)

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        x = self.lin0(x)

        for i in range(self.num_blocks):
            x = self.blocks[i](x, edge_index, edge_attr)

        x = self.lin1(x)
        #x = torch.sigmoid(x)
        return x
    

###############################################################################################
class GATResMeanConvWithRemask(torch.nn.Module):
    def __init__(self, name='GATResMeanConvWithRemask', num_blocks=5, nc = 32):
        super(GATResMeanConvWithRemask, self).__init__()

        self.num_blocks = num_blocks

        self.encoder = Linear(1, nc)
        self.blocks = torch.nn.ModuleList()
        self.name = name
        for _ in range(self.num_blocks):
            block = GResBlockMeanConv(nc, nc, nc)
            self.blocks.append(block)

        self.decoder = Linear(nc, 1)

        

    def forward(self, x : torch.Tensor, edge_index, batch=None, edge_attr=None, batch_mask=None, batch_second_mask=None):
        assert batch_mask is not None, 'input batch mask for Remasking strategy'
        
        batch_unmask = ~batch_mask.bool()
        #x has shape (bn, 1)
        #unmasked_x has shape (bn, 1)
        unmasked_x = x[batch_unmask] 

        #unmasked_x has shape (bn, nc)
        unmasked_x = self.encoder(unmasked_x)

        #x has shape (bn, nc)
        x = x.repeat(1,unmasked_x.size(-1))
        x[batch_unmask] = unmasked_x

        #x has shape (bn, nc)
        for i in range(self.num_blocks):
            x = self.blocks[i](x, edge_index, edge_attr)

        #remask strategy
        #if batch_second_mask is not None and self.training:
        #    x[batch_second_mask] = 0.0

        x = self.decoder(x)
        return x
    

###############################################################################################
import torch_geometric.utils as pgu
from torch.nn import Parameter

from torch_geometric.nn.conv import SimpleConv, GCNConv

class GResBlockConv(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hc):
        super(GResBlockConv, self).__init__()

        self.conv1 = GATConv(in_dim, hc, 2, concat=True)
        self.conv2 = GATConv(hc * 2, out_dim, 1, concat=False)

    def forward(self, x, edge_index, edge_attr):
        x_0 = torch.clone(x)
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x + x_0
        x = F.relu(x)
        return x

class GATResMeanConvWithRemaskAndStack(torch.nn.Module):
    def __init__(self, name='GATResMeanConvWithRemaskAndStack', num_blocks=5, nc = 32):
        super(GATResMeanConvWithRemaskAndStack, self).__init__()

        self.num_blocks = num_blocks

        self.encoder = Linear(1, nc)
        self.steam = GCNConv(1, nc, normalize=False)
        self.blocks = torch.nn.ModuleList()
        self.name = name
        for _ in range(self.num_blocks):
            block = GResBlockConv(nc, nc, nc)
            self.blocks.append(block)
        self.mask_token = torch.nn.Parameter(torch.zeros( 1, nc),False)
        self.decoder = Linear(nc, 1)


    def forward(self, x : torch.Tensor, edge_index, batch=None, edge_attr=None, batch_mask=None):
        assert batch_mask is not None, 'input batch mask for Remasking strategy'

        batch_unmask = ~batch_mask.bool()
        #x has shape (bn, 1)
        #unmasked_x has shape (len_unmask, 1)
        unmasked_x = x[batch_unmask]

        #unmasked_x has shape (len_unmask, nc)
        unmasked_x = self.encoder(unmasked_x)

        #gap_unmasked_x has shape (1, nc)
        gap_unmasked_x = unmasked_x.mean(dim=0,keepdim=True)
        
        #x has shape (bn, nc)
        x = self.steam(x, edge_index)
        
        x = x + gap_unmasked_x
        
        #x has shape (bn, nc)
        for i in range(self.num_blocks):
            #x has shape (bn, nc)->(bn, nc*2)
            x = self.blocks[i](x, edge_index, edge_attr)

        x = self.decoder(x)
        return x