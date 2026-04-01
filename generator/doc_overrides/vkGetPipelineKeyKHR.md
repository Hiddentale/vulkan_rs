# Usage Notes

Computes a pipeline key that identifies the pipeline configuration
for use with pipeline binaries. The key is a hash of the pipeline
create info that lets you look up previously cached binaries.

Pass a `PipelineCreateInfoKHR` referencing the pipeline create
info (graphics, compute, or ray tracing) to get its key. Pass
`None` to get an empty key structure for use as an output
parameter.

Store the key alongside serialized binary data from
`get_pipeline_binary_data_khr`. On subsequent runs, compute the
key for the current pipeline configuration and check if a matching
binary exists before falling back to full compilation.
