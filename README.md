# Fake bootstrapping example

This repository shows how to bootstrap FAKE via paket clitool.
You get a running fake via:

- `dotnet build` (which will run `.paket/paket.exe restore`)
- Or alternatively `dotnet restore` followed by `dotnet fake build`

The version of dotnet-fake is managed via `paket.dependencies` and `paket.lock`.

To upgrade to latest packages, just use `.paket/paket.exe update` and run fake again.