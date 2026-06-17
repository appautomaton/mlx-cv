You are reviewing the mlx-cv Depth Anything 3 parity work as an external engineering reviewer.

Model/run context:
- You are Antigravity CLI using Gemini 3.5 Flash (High).
- Workspace: /Users/ac/dev/ai/mlx-cv.
- This is a READ-ONLY review and diagnosis task.
- Do NOT modify any files.
- Do NOT run commands that write source code.
- If you need to propose changes, describe them as guidance only, with file/function names and rationale.
- Hand off cleanly to Codex: give a ranked diagnosis, likely root causes, concrete next probes, and exact files/functions to inspect or edit.

Concern to review:
We previously claimed DA3-SMALL multi-view parity was verified using a synthetic 3-view fixture. A later real-image smoke check showed the evidence is not sufficient:

1. Real 2-image SOH demo:
- Input images:
  - references/Depth-Anything-3/assets/examples/SOH/000.png
  - references/Depth-Anything-3/assets/examples/SOH/010.png
- Artifact summary:
  - /tmp/mlx-cv-da3-real-demo/parity_summary.json
- Results:
  - depth max_abs_error 0.0466212034, tolerance 0.05, passed
  - confidence max_abs_error 0.2812455893, tolerance 0.05, failed
  - extrinsics max_abs_error 0.0229539052, tolerance 0.15, passed
  - intrinsics max_abs_error 9.5165100098, tolerance 12.0, passed

2. Real 3-frame video demo:
- Input frames extracted from references/Depth-Anything-3/assets/examples/robot_unitree.mp4:
  - /tmp/mlx-cv-da3-real-video-frames/frame_0000.png
  - /tmp/mlx-cv-da3-real-video-frames/frame_0087.png
  - /tmp/mlx-cv-da3-real-video-frames/frame_0170.png
- Artifact summary:
  - /tmp/mlx-cv-da3-real-video-demo/parity_summary.json
  - /tmp/mlx-cv-da3-real-video-demo/da3_contact_sheet.png
- Results:
  - depth max_abs_error 0.0161633492, tolerance 0.05, passed
  - confidence max_abs_error 0.1753456593, tolerance 0.05, failed
  - extrinsics max_abs_error 0.0151755819, tolerance 0.15, passed
  - intrinsics max_abs_error 12.3303222656, tolerance 12.0, failed slightly
  - selected taps passed under the existing broad tolerances

Relevant files to inspect:
- tools/da3_demo.py
- tools/da3_upstream.py
- tools/da3_convert_checkpoint.py
- tools/da3_real_architecture_contract.py
- src/mlx_cv/parity/da3_real.py
- src/mlx_cv/parity/fixtures.py
- src/mlx_cv/models/depth_anything_v3/modeling.py
- src/mlx_cv/models/depth_anything_v3/processor.py
- src/mlx_cv/models/depth_anything_v3/camera.py
- src/mlx_cv/models/depth_anything_v3/convert.py
- src/mlx_cv/heads/dense/dualdpt.py
- src/mlx_cv/heads/dense/dpt.py
- src/mlx_cv/heads/dense/convert.py
- references/Depth-Anything-3/src/depth_anything_3/api.py
- references/Depth-Anything-3/src/depth_anything_3/model/da3.py
- references/Depth-Anything-3/src/depth_anything_3/model/dualdpt.py
- references/Depth-Anything-3/src/depth_anything_3/model/dpt.py
- references/Depth-Anything-3/src/depth_anything_3/model/cam_dec.py
- references/Depth-Anything-3/src/depth_anything_3/model/utils/head_utils.py
- references/Depth-Anything-3/src/depth_anything_3/utils/io/input_processor.py
- references/Depth-Anything-3/src/depth_anything_3/utils/io/output_processor.py
- tests/test_da3_upstream_parity.py
- tests/test_da3_real_forward.py
- tests/test_da3_real_checkpoint_load.py
- tests/test_da3_real_architecture_contract.py
- tests/test_da3_multiview_model.py

Known local changes since the previous synthetic parity work:
- tools/da3_demo.py now supports --image, writes input PNGs, contact sheet, README.
- tools/da3_upstream.py now permits at least two square RGB views instead of exactly three.
- tests/test_da3_upstream_parity.py now asserts real image loading and the new demo artifacts.

Important constraint:
Do not simply say "tolerance should be loosened." Treat confidence drift as real until proven otherwise. Determine whether the drift likely comes from preprocessing, output extraction, head logits/channel layout/activation, MLX-vs-Torch operation semantics, conversion/transposition, camera decoder/fov math, reference-view selection, or test/demo setup.

Questions to answer:
1. Based on code inspection, what are the most likely root causes of the confidence drift?
2. Is the intrinsics drift likely caused by the same issue as confidence, or separate camera decoder/fov math?
3. Are the current selected tap comparisons sufficient? If not, which additional taps should Codex compare to localize the drift?
4. What exact next diagnostic probes should Codex run, in order?
5. What code areas are most suspect, and what correction would you try first?
6. Is it defensible to mark DA3 phase as verified right now? If not, what wording/status should the roadmap use?

Expected output:
- Verdict: APPROVED / APPROVED_WITH_RISK / CHANGES_NEEDED.
- Top findings ranked by severity.
- Likely root cause hypotheses with evidence from file/function names.
- Step-by-step handoff plan for Codex.
- No file edits. No patches. No source modifications.
