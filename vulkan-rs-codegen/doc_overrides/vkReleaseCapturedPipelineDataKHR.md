# Usage Notes

Releases internal data captured during pipeline creation that was
retained for binary extraction. Call this after you have finished
calling `create_pipeline_binaries_khr` for a pipeline created
with `PIPELINE_CREATE_CAPTURE_DATA`.

The pipeline itself remains valid after this call, only the
captured internal data is freed. This reduces memory usage when
you no longer need to extract binaries from the pipeline.
