# Usage Notes

Destroys a pipeline binary handle. After destruction, the binary
cannot be used to create pipelines or retrieve data.

Pipeline binaries are independent of the pipelines created from
them, destroying a binary does not affect any pipeline that was
already created using it.
