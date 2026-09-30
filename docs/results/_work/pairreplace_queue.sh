#!/bin/bash
# Train PixArt with the injection mechanism replaced by rewriting the DiT's own
# cross-attention weights at the parsed object->attribute pairs.
#
#   w = softmax(DiT logits);  new_a = w @ edge;  w[attr] = new_a;  renorm
#
# so an attribute token carries the noun's own weight.  Config:
# configs/PixArt_xl2_coco2017_pair_replace.py
set -u
cd /root/private_data/PixArt-alpha-attentiongate
L=/root/compbench_work/logs

source /root/private_data/PixArt-alpha-attentiongate/results/_work/queue_wait.sh
wait_for_new_marker "$L/ditpair_queue.log" "DITPAIR_DONE" "[d]itpair_queue.sh"
echo "########## PAIR-REPLACE TRAIN start $(date -Is) ##########"
sleep 60

CK=output/coco2017_pair_replace/checkpoints/epoch_1_step_14786.pth
PYTHONPATH=. accelerate launch --num_processes 1 train_scripts/train.py \
    configs/PixArt_xl2_coco2017_pair_replace.py
echo "########## TRAIN rc=$? $(date -Is) ##########"
if [ ! -f "$CK" ]; then echo "ABORT: no checkpoint $(date -Is)"; exit 1; fi

echo "########## COMPBENCH GENERATE (7 categories, 1 img/prompt) $(date -Is) ##########"
python3 -u results/_work/gen_compbench.py --method pair_replace --batch-size 16 \
    --categories color,shape,texture,spatial,3d_spatial,numeracy,non_spatial \
    --start-repeat 0 --end-repeat 1 2>&1 \
  | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"

# complex is scored by 3_in_1.py, which hardcodes num=10 images per prompt, so it
# requires the 100-prompt x 10-image protocol tpscda/frozen were scored under
# (see run_complex_gen.sh).  At 1 image/prompt the metric cannot be computed at
# all -- that is why gateopt/b2/b2_neutral/nodistill have no complex score.
echo "########## COMPBENCH GENERATE complex (100 prompts x 10) $(date -Is) ##########"
python3 -u results/_work/gen_compbench.py --method pair_replace --batch-size 16 \
    --categories complex --limit-prompts 100 --start-repeat 0 --end-repeat 10 2>&1 \
  | grep -vE "it/s\]|FutureWarning|warnings.warn|Loading checkpoint shards"

echo "########## COMPBENCH EVALUATE $(date -Is) ##########"
cd results/_work
for c in color shape texture spatial 3d_spatial numeracy non_spatial complex; do
  echo "----- pair_replace / $c $(date -Is) -----"
  timeout 10800 python3 run_eval.py --method pair_replace --category "$c" \
      --max-batch 8 --blip-max-batch 32 --min-free-gb 4 >> "$L/eval_pairreplace_${c}.log" 2>&1
  echo "  rc=$? $(date -Is)"
done
cd /root/private_data/PixArt-alpha-attentiongate

echo "########## RESULT $(date -Is) ##########"
python3 results/_work/compbench_paired_test.py --baseline frozen --methods pair_replace 2>&1 | tail -11
python3 results/_work/compbench_paired_test.py --baseline tpscda --methods pair_replace 2>&1 | tail -11
echo "PAIRREPLACE_DONE $(date -Is)"
