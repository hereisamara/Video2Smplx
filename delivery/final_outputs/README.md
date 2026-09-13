# Final Output Drop Folder

Put completed run outputs here before building the final document set or flash
drive set.

Recommended structure:

```text
delivery/final_outputs/
  SignLanguage_S2/
    smplx_params.npz
    smplx_render.mp4
    side_by_side_input_render.mp4
    combine_render_report.json
```

Use `delivery/scripts/collect_server_outputs_example.sh` as a template for
copying generated outputs from the server.
