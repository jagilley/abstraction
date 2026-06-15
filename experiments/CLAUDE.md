# Claude Code - experiment-specific guidance

- Please use `modal run --detach` for any Modal jobs that we expect to take longer than 2 minutes. It can be easy to kill jobs when running in attached mode.
- You need to run detached functions by invoking the function explicitly with e.g. `modal run --detach a2a_forward/permutation_test.py::a2a_permutation_test` rather than just `modal run --detach a2a_forward/permutation_test.py`. The `--detach` parameter only persists the most recently-created function, and normally you can't really control the order they get created in. So it's best to invoke explicitly.
- Likewise, running e.g. `modal run --detach a2a_forward/permutation_test.py::main` as a local entrypoint will also result in premature cancellations, so don't do this.