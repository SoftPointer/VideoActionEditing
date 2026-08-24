#!/usr/bin/env bash
set -euo pipefail

# Review-media generation only.  This launcher performs no training and creates
# four source-free T2V positives per complex event.  Noop/reverse/incomplete are
# deterministic frame transforms of each positive, hence share exact appearance.
if [ "$#" -ne 1 ]; then echo "usage: $0 GPU_GROUP" >&2; exit 2; fi
case "$1" in 0) devices=0,1,2,3 ;; 1) devices=4,5,6,7 ;; *) exit 2 ;; esac
case "$(hostname -s):$1" in
  auh7-1b-gpu-246:0) shard=0 ;; auh7-1b-gpu-246:1) shard=1 ;;
  auh7-1b-gpu-247:0) shard=2 ;; auh7-1b-gpu-247:1) shard=3 ;;
  auh7-1b-gpu-248:0) shard=4 ;; auh7-1b-gpu-248:1) shard=5 ;;
  auh7-1b-gpu-279:0) shard=6 ;; auh7-1b-gpu-279:1) shard=7 ;;
  *) echo "forbidden node/group outside Job 140846" >&2; exit 3 ;;
esac

root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/action_flow_noise_stage0_job140846_v1
runtime_tree="$root/stage1/source-be31323"
manifest="$root/stage1/interaction_complex8_multianchor_authoring_v2.json"
output_root="$root/stage1/interaction_complex8_multianchor_v2_r1"
python_bin=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
bernini_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591
veomni_root=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11
checkpoint=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4
checkpoint_manifest=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_appearance_counterfactual_20260808_74ed30c/runtime/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256
runner="$runtime_tree/methods/bernini_action_editing/infer_native_identity_generation_canary.py"

test -f "$manifest"
test -f "$runner"
test "$(jq -r .schema_version "$manifest")" = bernini-interaction-complex8-multianchor-authoring-v2
test "$(jq -r .training_authorized_before_video_review "$manifest")" = false
test "$(jq -r .qwen_used "$manifest")" = false
test "$(jq '[.events[] | .variants[]] | length' "$manifest")" -eq 32
mkdir -p "$output_root"

for index in $(seq "$shard" 8 31); do
  row="$(jq -c --argjson n "$index" '[.events[] as $event | $event.variants[] | {ordinal:$event.ordinal,source_iid:$event.source_iid,event_id:$event.event_id,category:$event.category,review_requirement:$event.review_requirement,source_video:$event.geometry_source_video,action:$event.action,constraints:$event.constraints,variant_id:.variant_id,seed:.seed,setup:.setup}] | .[$n]' "$manifest")"
  test "$row" != null
  ordinal="$(jq -r .ordinal <<<"$row")"
  source_iid="$(jq -r .source_iid <<<"$row")"
  event_id="$(jq -r .event_id <<<"$row")"
  variant_id="$(jq -r .variant_id <<<"$row")"
  source_video="$(jq -r .source_video <<<"$row")"
  seed="$(jq -r .seed <<<"$row")"
  prompt="$(jq -r '[.setup,.action,.constraints] | join(" ")' <<<"$row")"
  output="$output_root/e$(printf '%02d' "$ordinal")_${event_id}/$variant_id"
  if [ -f "$output/REVIEW_MEDIA_COMPLETE" ]; then
    echo "already complete: $output"
    continue
  fi
  test -f "$source_video"
  source_sha="$(sha256sum "$source_video" | awk '{print $1}')"
  prompt_sha="$(printf %s "$prompt" | sha256sum | awk '{print $1}')"

  scratch="/tmp/interaction-complex8-v2-${index}-140846"
  mkdir -p "$scratch/miopen-user" "$scratch/miopen-custom" "$scratch/torch-extensions" "$scratch/triton"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  export MODELING_BACKEND=hf OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export ROCR_VISIBLE_DEVICES="$devices"
  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  export MIOPEN_USER_DB_PATH="$scratch/miopen-user" MIOPEN_CUSTOM_CACHE_DIR="$scratch/miopen-custom"
  export TORCH_EXTENSIONS_DIR="$scratch/torch-extensions" TRITON_CACHE_DIR="$scratch/triton"
  export PYTHONPATH="$runtime_tree/methods/bernini_action_editing"

  if [ ! -e "$output" ]; then
    mkdir -p "$(dirname "$output")"
    "$python_bin" -B -m torch.distributed.run --standalone --nproc_per_node=4 \
      "$runner" \
      --bernini-root "$bernini_root" \
      --veomni-root "$veomni_root" \
      --checkpoint "$checkpoint" \
      --checkpoint-content-manifest "$checkpoint_manifest" \
      --source-video "$source_video" \
      --expected-source-sha256 "$source_sha" \
      --action-prompt "$prompt" \
      --expected-action-prompt-sha256 "$prompt_sha" \
      --output-dir "$output" \
      --arms t2v \
      --num-inference-steps 40 \
      --seed "$seed" \
      --method-source-revision be3132312b77125313901c928b7aedcfc2c72c12 \
      --method-source-archive-sha256 958b9350e32b5459053a4aa62dff6334fb0b251f41e8e60dcd16643cef0f9d3e
  else
    # A model-complete directory without the review marker is a recoverable
    # post-processing checkpoint (for example, ffmpeg was unavailable).  It
    # may be resumed only when the two immutable native artifacts exist and no
    # derived review artifact has been published yet.
    test -d "$output"
    test -f "$output/t2v.mp4"
    test -f "$output/receipt.json"
    test ! -e "$output/review_receipt.json"
    test ! -e "$output/noop.mp4"
    test ! -e "$output/reverse.mp4"
    test ! -e "$output/incomplete.mp4"
    echo "resume post-processing: $output"
  fi

  test -f "$output/t2v.mp4"
  test -f "$output/receipt.json"

  "$python_bin" -B - "$output" "$row" "$prompt" "$prompt_sha" "$source_sha" "$index" <<'PY'
from fractions import Fraction
import hashlib, json, os, pathlib, sys
import av
import numpy as np
out=pathlib.Path(sys.argv[1]); row=json.loads(sys.argv[2]); prompt=sys.argv[3]
prompt_sha, source_sha, index=sys.argv[4],sys.argv[5],int(sys.argv[6])
native=json.loads((out/'receipt.json').read_text())
assert native['arms']==['t2v']
assert native['input']['external_first_frame_anchor'] is False
assert native['input']['target_video'] is False
assert native['conditioning']['t2v']['full_source_video_count']==0
assert native['conditioning']['t2v']['source_derived_reference_count']==0
assert native['freeze_certificate']['trainable_parameter_elements']==0
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def decode(path):
    with av.open(str(path),'r') as container:
        streams=list(container.streams.video)
        assert len(streams)==1 and streams[0].average_rate==25
        frames=[np.ascontiguousarray(frame.to_ndarray(format='rgb24')) for frame in container.decode(streams[0])]
    assert len(frames)==81 and len({frame.shape for frame in frames})==1
    return frames
def encode(path,frames):
    assert len(frames)==81 and not path.exists()
    temporary=path.with_name(path.name+'.partial')
    assert not temporary.exists()
    height,width=frames[0].shape[:2]
    with av.open(str(temporary),'w',format='mp4') as container:
        stream=container.add_stream('libx264',rate=25,options={'crf':'18','preset':'medium'})
        stream.width=width; stream.height=height; stream.pix_fmt='yuv420p'; stream.time_base=Fraction(1,25)
        for ordinal,array in enumerate(frames):
            frame=av.VideoFrame.from_ndarray(array,format='rgb24'); frame.pts=ordinal; frame.time_base=Fraction(1,25)
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    os.replace(temporary,path)
def probe(path):
    with av.open(str(path),'r') as container:
        streams=list(container.streams.video); assert len(streams)==1
        stream=streams[0]; count=sum(1 for _ in container.decode(stream))
        codec=stream.codec_context.name; rate=Fraction(stream.average_rate)
        width=int(stream.codec_context.width); height=int(stream.codec_context.height)
    assert codec=='h264' and rate==Fraction(25,1) and count==81
    return {'codec_name':codec,'avg_frame_rate':'25/1','nb_read_frames':count,'width':width,'height':height}
action_frames=decode(out/'t2v.mp4')
encode(out/'noop.mp4',[action_frames[0]]*81)
encode(out/'reverse.mp4',list(reversed(action_frames)))
encode(out/'incomplete.mp4',action_frames[:41]+[action_frames[40]]*40)
media={}
for role,name in [('action','t2v.mp4'),('noop','noop.mp4'),('reverse','reverse.mp4'),('incomplete','incomplete.mp4')]:
    path=out/name; media[role]={'file':name,'sha256':sha(path),'probe':probe(path)}
receipt={'schema_version':'bernini-interaction-complex8-review-media-receipt-v2','complete':True,'global_index':index,'event_ordinal':row['ordinal'],'source_iid':row['source_iid'],'event_id':row['event_id'],'category':row['category'],'variant_id':row['variant_id'],'seed':row['seed'],'prompt':prompt,'prompt_sha256':prompt_sha,'geometry_source_sha256':source_sha,'lineage':'source_free_pure_t2v','training_performed':False,'qwen_used':False,'negative_construction':{'noop':'repeat_frame_0_exact81','reverse':'reverse_exact81','incomplete':'frames_0_40_then_hold_frame_40_to_80'},'media':media}
(out/'review_receipt.json').write_text(json.dumps(receipt,sort_keys=True,separators=(',',':'))+'\n')
PY
  touch "$output/REVIEW_MEDIA_COMPLETE"
done

touch "$output_root/SHARD_${shard}_COMPLETE"
