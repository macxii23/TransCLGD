import torch #compose Linear(+PE) encoders; Interchange encoders register themselves in interchange_encoder.py
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder, register_edge_encoder
from lgd.encoder.linear_node_encoder import LinearNodeEncoder
from lgd.encoder.linear_edge_encoder import LinearEdgeEncoder
from lgd.encoder.dummy_edge_encoder import DummyEdgeEncoder


def concat_node_encoders(encoder_classes, pe_enc_names, edge=False): #factory: concat dataset encoder + optional PE encoders
    class Concat2NodeEncoder(torch.nn.Module):
        enc1_cls = None; enc2_cls = None; enc2_name = None; edge = False
        def __init__(self, dim_emb):
            super().__init__()
            enc2_dim_pe = getattr(cfg, f"posenc_{self.enc2_name}").dim_pe
            self.encoder1 = self.enc1_cls(dim_emb - enc2_dim_pe)
            self.encoder2 = self.enc2_cls(dim_emb, expand_x=False)
        def forward(self, batch):
            return self.encoder2(self.encoder1(batch))

    class Concat3NodeEncoder(torch.nn.Module):
        enc1_cls = None; enc2_cls = None; enc2_name = None; enc3_cls = None; enc3_name = None
        def __init__(self, dim_emb):
            super().__init__()
            enc2_dim_pe = getattr(cfg, f"posenc_{self.enc2_name}").dim_pe
            enc3_dim_pe = getattr(cfg, f"posenc_{self.enc3_name}").dim_pe
            self.encoder1 = self.enc1_cls(dim_emb - enc2_dim_pe - enc3_dim_pe)
            self.encoder2 = self.enc2_cls(dim_emb - enc3_dim_pe, expand_x=False)
            self.encoder3 = self.enc3_cls(dim_emb, expand_x=False)
        def forward(self, batch):
            return self.encoder3(self.encoder2(self.encoder1(batch)))

    if len(encoder_classes) == 2:
        Concat2NodeEncoder.enc1_cls, Concat2NodeEncoder.enc2_cls = encoder_classes
        Concat2NodeEncoder.enc2_name, Concat2NodeEncoder.edge = pe_enc_names[0], edge
        return Concat2NodeEncoder
    if len(encoder_classes) == 3:
        Concat3NodeEncoder.enc1_cls, Concat3NodeEncoder.enc2_cls, Concat3NodeEncoder.enc3_cls = encoder_classes
        Concat3NodeEncoder.enc2_name, Concat3NodeEncoder.enc3_name = pe_enc_names
        return Concat3NodeEncoder
    raise ValueError(f"Does not support concatenation of {len(encoder_classes)} encoder classes.")


ds_encs = {'LinearNode': LinearNodeEncoder}
ds_edge_encs = {'LinearEdge': LinearEdgeEncoder, 'DummyEdge': DummyEdgeEncoder}
# PE-composed Linear* names are unused by Interchange YAML (uses Interchange*); keep Linear/Dummy registered via their modules.
_ = (register_node_encoder, register_edge_encoder, concat_node_encoders, ds_encs, ds_edge_encs, cfg)
