# Final Output Drop Folder

Put completed and verified run outputs here before building the final document
set or flash-drive set.

Required structure:

```text
delivery/final_outputs/
  SignLanguage_S2/
    smplx_params.npz
    rendered/
      smplx_render.mp4
    side_by_side_input_render.mp4
    combine_render_report.json
    runtime_report.json
    geometry_summary.csv
```

On the server, use `delivery/scripts/install_server_delivery_assets.sh` to copy
the verified final run into this folder together with the learned corrector
checkpoints. Then run `delivery/scripts/verify_delivery_readiness.sh`.

Alternatively, use `delivery/scripts/collect_server_outputs_example.sh` from a
workstation to download a completed run. The document-set and flash-drive build
scripts reject incomplete final-output folders by default.
