"""TP-SCDA + B1: pair-aware bias follows the PARSED object->attribute edges.

Rationale (results/TP_SCDA_IMPROVEMENT_PLAN.md section B1): the pair affinity
was driven purely by a learned content similarity, discarding the dependency
structure the parser already provides (0.00% alignment failure rate on this
corpus).  B1 multiplies the affinity by the parsed edge adjacency, so an object
can only reinforce the attribute it is actually bound to.

Keeps every gate control from the gate-optimisation run, so the only new
variable relative to that run is the edge gate.
"""
exec(compile(open('/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_token_pair_gate_opt.py', encoding='utf-8').read(),
             '/root/private_data/PixArt-alpha-attentiongate/configs/PixArt_xl2_coco2017_token_pair_gate_opt.py', 'exec'))

experiment_name = 'coco2017_token_pair_b1_edgegate'
work_dir = '/root/private_data/PixArt-alpha-attentiongate/output/coco2017_token_pair_b1_edgegate'

# --- B1 ---
semantic_pair_edge_gate = True
semantic_binding_coef = 0.0

# parsed edges for the dataset
data = dict(data, load_semantic_edges=True,
            semantic_edge_index='/root/private_data/data/COCO2017Prepared/partition/edge_index.npz', max_semantic_edges=24)

num_epochs = 1
max_train_steps = 0
save_model_steps = 7000
