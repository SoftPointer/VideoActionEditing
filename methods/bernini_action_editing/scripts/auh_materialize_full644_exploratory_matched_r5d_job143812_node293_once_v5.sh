#!/bin/bash
# One-shot handoff for the fresh r4 package from its sealed 21-file snapshot.
#
# This controller is intended to be supplied on stdin to exactly:
#   /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C \
#     HOME=/vast/users/guangyi.chen BASH_ENV=/dev/null \
#     /bin/bash -p -s
# on the AUH login node.  It creates no Slurm step.  Its Python preflight
# authenticates every materializer input before the materializer can perform
# the first mkdir.  The materializer itself is captured once through its held
# descriptor and is executed as those captured bytes by a held VACE Python.

set -Eeuo pipefail
umask 077

[[ "$0" == /bin/bash && "$#" -eq 0 && "$-" == *p* && "$-" != *i* ]] || {
  /usr/bin/printf 'r5d materialization refused: shell entry differs\n' >&2
  exit 96
}
[[ "${PATH:-}" == /usr/bin:/bin && "${LC_ALL:-}" == C \
  && "${LANG:-}" == C && "${HOME:-}" == /vast/users/guangyi.chen \
  && "${BASH_ENV:-}" == /dev/null ]] || {
  /usr/bin/printf 'r5d materialization refused: entry environment differs\n' >&2
  exit 96
}
[[ -z "${ENV:-}" && -z "${LD_PRELOAD:-}" && -z "${LD_LIBRARY_PATH:-}" \
  && -z "${PYTHONPATH:-}" && -z "${PYTHONHOME:-}" \
  && -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${ROCR_VISIBLE_DEVICES:-}" \
  && -z "${HIP_VISIBLE_DEVICES:-}" && -z "${GPU_DEVICE_ORDINAL:-}" ]] || {
  /usr/bin/printf 'r5d materialization refused: loader or GPU environment differs\n' >&2
  exit 96
}
if builtin declare -F | /usr/bin/grep . >/dev/null; then
  /usr/bin/printf 'r5d materialization refused: preloaded Bash function exists\n' >&2
  exit 96
fi
if shopt -q varredir_close 2>/dev/null; then
  shopt -u varredir_close
fi

readonly R5D_JOB_ID=143812
readonly R5D_NODE=auh7-1b-gpu-293
readonly R5D_SOURCE_ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_exploratory_matched_r5d_source_snapshot_21_20260820_r3
readonly R5D_TARGET_ROOT=/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_exploratory_matched_eval_auh_r5d_job143812_node293_case00_847b91a2_c91de7eb_85ccc17b_r4
readonly R5D_PYTHON=/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12
readonly R5D_MATERIALIZER="${R5D_SOURCE_ROOT}/methods/bernini_action_editing/tools/materialize_full644_exploratory_matched_r5d_case00_package_v4.py"

[[ ! -e "${R5D_TARGET_ROOT}" && ! -L "${R5D_TARGET_ROOT}" ]] || {
  /usr/bin/printf 'r5d materialization refused: fresh target already exists\n' >&2
  exit 96
}
[[ -f "${R5D_PYTHON}" && -x "${R5D_PYTHON}" && ! -L "${R5D_PYTHON}" \
  && -f "${R5D_MATERIALIZER}" && ! -L "${R5D_MATERIALIZER}" ]] || {
  /usr/bin/printf 'r5d materialization refused: held entry file differs\n' >&2
  exit 96
}

exec {R5D_PYTHON_FD}<"${R5D_PYTHON}"
exec {R5D_MATERIALIZER_FD}<"${R5D_MATERIALIZER}"
[[ "${R5D_PYTHON_FD}" =~ ^[0-9]+$ \
  && "${R5D_MATERIALIZER_FD}" =~ ^[0-9]+$ \
  && "${R5D_PYTHON_FD}" -ge 3 \
  && "${R5D_MATERIALIZER_FD}" -ge 3 \
  && "${R5D_PYTHON_FD}" != "${R5D_MATERIALIZER_FD}" ]] || {
  /usr/bin/printf 'r5d materialization refused: held descriptor allocation differs\n' >&2
  exit 96
}

R5D_BOOTSTRAP='import hashlib,os,stat,sys

def fail(message):
    raise RuntimeError(message)

def ident(info):
    return (info.st_dev,info.st_ino,info.st_uid,info.st_gid,info.st_mode,info.st_nlink,info.st_rdev,info.st_size,getattr(info,"st_blocks",0),info.st_mtime_ns,info.st_ctime_ns)

def read_fd(descriptor,size):
    blocks=[]
    offset=0
    while offset<size:
        block=os.pread(descriptor,min(1048576,size-offset),offset)
        if not block:
            break
        blocks.append(block)
        offset+=len(block)
    raw=b"".join(blocks)
    if len(raw)!=size:
        fail("held read size differs")
    return raw

def stable_fd(descriptor,path,digest,mode,executable=False,size=None):
    before=os.fstat(descriptor)
    first=read_fd(descriptor,before.st_size)
    middle=os.fstat(descriptor)
    second=read_fd(descriptor,before.st_size)
    after=os.fstat(descriptor)
    named=os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_uid!=2012 or before.st_gid!=2000:
        fail("held file topology or owner differs: "+path)
    observed_mode=stat.S_IMODE(before.st_mode)
    if mode is not None and observed_mode!=mode:
        fail("held file mode differs: "+path)
    if executable and (observed_mode&0o111)==0:
        fail("held executable mode differs: "+path)
    if size is not None and before.st_size!=size:
        fail("held file size differs: "+path)
    if ident(before)!=ident(middle) or ident(before)!=ident(after) or ident(before)!=ident(named):
        fail("held file identity differs: "+path)
    if first!=second or hashlib.sha256(first).hexdigest()!=digest:
        fail("held file bytes differ: "+path)
    return first

def stable_named(path,digest,kind):
    if not os.path.isabs(path) or os.path.normpath(path)!=path or os.path.islink(path):
        fail("named path differs: "+path)
    descriptor=os.open(path,os.O_RDONLY|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0))
    try:
        before=os.fstat(descriptor)
        first=read_fd(descriptor,before.st_size)
        middle=os.fstat(descriptor)
        second=read_fd(descriptor,before.st_size)
        after=os.fstat(descriptor)
        named=os.lstat(path)
    finally:
        os.close(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_uid!=2012 or before.st_gid!=2000:
        fail("source topology or owner differs: "+path)
    mode=stat.S_IMODE(before.st_mode)
    if kind=="source" and mode!=0o644:
        fail("source mode differs: "+path)
    if kind=="sealed" and (mode&0o022)!=0:
        fail("sealed input is group/world writable: "+path)
    if kind=="executable" and ((mode&0o111)==0 or (mode&0o022)!=0):
        fail("executable mode differs: "+path)
    if ident(before)!=ident(middle) or ident(before)!=ident(after) or ident(before)!=ident(named):
        fail("source identity differs: "+path)
    if first!=second or hashlib.sha256(first).hexdigest()!=digest:
        fail("source bytes differ: "+path)

if len(sys.argv)!=7:
    fail("bootstrap argv count differs")
python_fd_raw,materializer_fd_raw,python_path,materializer_path,source_root,target_root=sys.argv[1:]
try:
    python_fd=int(python_fd_raw)
    materializer_fd=int(materializer_fd_raw)
except ValueError as error:
    raise RuntimeError("bootstrap descriptor syntax differs") from error
if python_fd<3 or materializer_fd<3 or python_fd==materializer_fd:
    fail("bootstrap descriptor values differ")
if not os.get_inheritable(python_fd) or not os.get_inheritable(materializer_fd):
    fail("bootstrap descriptor inheritance differs")
if sys.stdin.buffer.read(1)!=b"":
    fail("bootstrap stdin is not exhausted")
if set(os.environ) not in (set(),{"LC_CTYPE"}) or ("LC_CTYPE" in os.environ and os.environ["LC_CTYPE"]!="C.UTF-8"):
    fail("bootstrap environment differs")
if sys.platform!="linux" or not os.path.isdir("/proc/self/fd") or sys.flags.isolated!=1 or sys.flags.no_site!=1 or sys.flags.ignore_environment!=1 or not sys.dont_write_bytecode:
    fail("bootstrap isolated startup differs")
if "torch" in sys.modules or "sitecustomize" in sys.modules or "usercustomize" in sys.modules:
    fail("bootstrap module preload differs")
if os.geteuid()!=2012 or os.getegid()!=2000:
    fail("bootstrap owner authority differs")
if python_path!="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12" or materializer_path!=source_root+"/methods/bernini_action_editing/tools/materialize_full644_exploratory_matched_r5d_case00_package_v4.py":
    fail("bootstrap frozen entry paths differ")
if source_root!="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_exploratory_matched_r5d_source_snapshot_21_20260820_r3" or target_root!="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_exploratory_matched_eval_auh_r5d_job143812_node293_case00_847b91a2_c91de7eb_85ccc17b_r4":
    fail("bootstrap binding differs")
if os.path.lexists(target_root):
    fail("fresh target exists before source preflight")
for directory in (source_root,os.path.dirname(target_root)):
    info=os.lstat(directory)
    if os.path.realpath(directory)!=directory or not stat.S_ISDIR(info.st_mode) or info.st_uid!=2012 or info.st_gid!=2000 or (stat.S_IMODE(info.st_mode)&0o002)!=0:
        fail("authority directory differs: "+directory)

python_raw=stable_fd(python_fd,python_path,"8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a",0o755,True,31490256)
if ident(os.fstat(python_fd))!=ident(os.stat("/proc/self/exe")):
    fail("held Python process identity differs")

source_specs=(
    ("methods/action_editing_baselines/manifests/goku_legacy_shared8_exposure.json","953933f1161b6d62826d388ba5ed42e42792fbf5f2bdeea199c1eb13cd251b4a"),
    ("methods/action_editing_baselines/manifests/goku_legacy_heldout8_inputs.jsonl","c05c4e5b5bf85de882bde32c71a984d736247733e586ed91d40026b12aaaf701"),
    ("methods/bernini_action_editing/action_preservation_decoded_eval_model_authority_v2.py","b9457e434b8000e5368056c925edd0227b4dd3d8a439090494af088817d51ecf"),
    ("methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256","a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"),
    ("methods/bernini_action_editing/full644_exploratory_matched_eval_v1.py","d6ef0939a67598e66ccf2652d22520ae3a87a068789f70f921522ba86046138d"),
    ("methods/bernini_action_editing/full644_exploratory_matched_eval_v2.py","b675b84fd5f1b95a21f6454c9eb8c53b0965d7dbdd3fedf1ea92b6ad153ac982"),
    ("methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_v2.py","53b75aea4897a0ec5ad70c8ea2b2dd314b93d1331cf5e41d65c3b51339f4d4ca"),
    ("methods/bernini_action_editing/full644_exploratory_matched_infer_adapter_auh_r5d.py","5794e1f0e5ecb84ffdb37f618fe63696ee4f87176952ac083c8c91792a9d192a"),
    ("methods/bernini_action_editing/full644_exploratory_matched_runner_auh_r5.py","847b91a267fe55cfbfa793027548f82beb5ec9630efab329878576ae6c5a9223"),
    ("methods/bernini_action_editing/full644_exploratory_matched_spooled_launcher_auh_r5.py","cb201398940d59393fa58471dc2c3f9fdf001c7e881ec891ce892bb460cf01ba"),
    ("methods/bernini_action_editing/full644_exploratory_matched_spooled_launcher_auh_r5d.py","85ccc17b30d97a7bf048702cd8a8ed10c3421e01721902fea7db6242eac45753"),
    ("methods/bernini_action_editing/full644_exploratory_matched_torchrun_fd_bridge_v2.py","c91de7eb821a05c61f66349c02f9232ede27c49e54659f351f72930fb071d136"),
    ("methods/bernini_action_editing/infer_lora.py","acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553"),
    ("methods/bernini_action_editing/self_generated_action_preservation_v2.py","11bc0792174a60c2e449eb61ff8f81da97808e02ee2707b5c4f20ee2118f4b5c"),
    ("methods/bernini_action_editing/tools/build_renderer_dataset.py","afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5"),
    ("methods/bernini_action_editing/tools/materialize_vae.py","a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0"),
    ("methods/bernini_action_editing/train_lora.py","ead547b8309e1b5ae5c831444e9f5d1d8e1785fed5fe39cf7b97f13f82a9ce85"),
    ("methods/bernini_action_editing/full644_exploratory_matched_r5d_root_bootstrap_probe_runner_v1.py","e4890e5d45c6a3982bab03f311711effc87efd29718ff8d5726ad4580b8a3845"),
    ("methods/bernini_action_editing/full644_exploratory_matched_r5d_static_nomodel_probe_v3.py","c03a40b1e1853a76f33bd98ea2c96108a5c94cf02d6970e60d1dfcc22f2cd7b0"),
    ("methods/bernini_action_editing/full644_exploratory_matched_r5d_cpu_consumption_probe_v1.py","5c7f5caf5ad73aecacedda618e941308e4fc1b94218b71cdc44e88afc3d3f0ea"),
)
if len(source_specs)!=20 or len({row[0] for row in source_specs})!=20:
    fail("source pin closure differs")
for relative,digest in source_specs:
    stable_named(source_root+"/"+relative,digest,"source")

external_specs=(
    ("/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2","e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99","executable"),
    ("/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/torch/distributed/run.py","1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c","sealed"),
    ("/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py","9871ee801f346c4952fcaf2cc87965f3c997d974b550df70e1fc7f4534c66e87","sealed"),
    ("/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/torch/distributed/elastic/agent/server/local_elastic_agent.py","71f390071316417643aa91514ebb170b3adb7eca5c1fe8286d03fe2eef21e497","sealed"),
    ("/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/torch/distributed/elastic/rendezvous/dynamic_rendezvous.py","adc34f683614cdc6de5f5cc64e34ee7201b0671609a7ee574b9731f4266e5cec","sealed"),
    ("/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/api.py","f815c915fd857bbff12b4d00530c7c1ffb0badfcd48c41e7f378c65828192ef7","sealed"),
    ("/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/runtime/ffprobe_conda_cf_ffmpeg9_hdabad70_r2/runtime/ffprobe","356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5","executable"),
    ("/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_exploratory_r64_job141620_v5/runs/full644-r64-reference-dpo-preservation-one-pass-v5/checkpoint-00000644/checkpoint_manifest.json","7a4864a3ffa50c12af91f8d2b88610a6cd8f994aa68eef8d27b95bcc2d73d3b2","sealed"),
)
if len(external_specs)!=8 or len({row[0] for row in external_specs})!=8:
    fail("external pin closure differs")
for path,digest,kind in external_specs:
    stable_named(path,digest,kind)

for directory in (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4",
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/Bernini-2d2b4591",
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/phase_a_native_gpu_canary_dual4_all8_v1/releases/source-00f7aba-launcher-1dafc42-r1/vendor/VeOmni-f90b3dc6",
):
    info=os.lstat(directory)
    if os.path.realpath(directory)!=directory or not stat.S_ISDIR(info.st_mode) or info.st_uid!=2012 or info.st_gid!=2000 or (stat.S_IMODE(info.st_mode)&0o002)!=0:
        fail("runtime directory differs: "+directory)

materializer_raw=stable_fd(materializer_fd,materializer_path,"1609aac08610ab426679e8eba414a070d8a6ba9f7508dc0bd6fc9edd4379cbb0",0o644,False,36169)
try:
    materializer_source=materializer_raw.decode("utf-8","strict")
except UnicodeDecodeError as error:
    raise RuntimeError("materializer is not strict UTF-8") from error
compile(materializer_source,materializer_path,"exec",dont_inherit=True)
if os.path.lexists(target_root):
    fail("fresh target exists after complete preflight")
if ident(os.fstat(python_fd))!=ident(os.stat(python_path)) or ident(os.fstat(python_fd))!=ident(os.stat("/proc/self/exe")):
    fail("held Python changed across complete preflight")
if ident(os.fstat(materializer_fd))!=ident(os.lstat(materializer_path)) or hashlib.sha256(read_fd(materializer_fd,os.fstat(materializer_fd).st_size)).hexdigest()!="1609aac08610ab426679e8eba414a070d8a6ba9f7508dc0bd6fc9edd4379cbb0":
    fail("held materializer changed across complete preflight")
os.set_inheritable(materializer_fd,False)
os.set_inheritable(python_fd,True)
argv=[python_path,"-I","-S","-B","-c",materializer_source,"--job-id","143812","--node","auh7-1b-gpu-293","--source-root",source_root]
os.execve("/proc/self/fd/"+str(python_fd),argv,{})'
readonly R5D_BOOTSTRAP

exec -c "/proc/self/fd/${R5D_PYTHON_FD}" -I -S -B -c "${R5D_BOOTSTRAP}" \
  "${R5D_PYTHON_FD}" "${R5D_MATERIALIZER_FD}" \
  "${R5D_PYTHON}" "${R5D_MATERIALIZER}" \
  "${R5D_SOURCE_ROOT}" "${R5D_TARGET_ROOT}"
