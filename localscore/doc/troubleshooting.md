# Troubleshooting the LocalScore CLI

Much of the troubleshooting for LocalScore is similar to Llamafile.

Some more common issues are pointed out below.

## CLI

### Windows

On Windows, there's a variety of issues.

#### Model Size

If you are using a Llamafile or LocalScore benchmark bundle larger than 4 gigabytes, you will not be able to run it on Windows due to Windows limitations. You can use LocalScore as a standalone utility and pass in models in GGUF format.

#### WSL2 (from LLamafile docs)

On WSL, there are many possible gotchas. One thing that helps solve them
completely is this:

```
[Unit]
Description=cosmopolitan APE binfmt service
After=wsl-binfmt.service

[Service]
Type=oneshot
ExecStart=/bin/sh -c "echo ':APE:M::MZqFpD::/usr/bin/ape:' >/proc/sys/fs/binfmt_misc/register"

[Install]
WantedBy=multi-user.target
```

Put that in `/etc/systemd/system/cosmo-binfmt.service`.

Ensure that the APE loader is installed to `/usr/bin/ape`:

```sh
sudo wget -O /usr/bin/ape https://cosmo.zip/pub/cosmos/bin/ape-$(uname -m).elf
sudo chmod +x /usr/bin/ape
```

Then run `sudo systemctl enable --now cosmo-binfmt`.

Another thing that's helped WSL users who experience issues, is to
disable the WIN32 interop feature:

```sh
sudo sh -c "echo -1 > /proc/sys/fs/binfmt_misc/WSLInterop"
```

In Windows 11 with WSL 2 the location of the interop flag has changed, as such
the following command be required instead/additionally:

```sh
sudo sh -c "echo -1 > /proc/sys/fs/binfmt_misc/WSLInterop-late"
```

In the instance of getting a `Permission Denied` on disabling interop
through CLI, it can be permanently disabled by adding the following in
`/etc/wsl.conf`

```sh
[interop]
enabled=false
```

#### Other Issues

We have observed that on Windows the performance of LocalScore is slower than on Linux. This is expected at the moment.

### Linux

On Linux, there are a few issues that have been observed.

### macOS

Right now, we don't know of any major issues on macOS. Please report and they will be added here.

Please file a GitHub issue if you encounter any problems. At the very least we can document troubleshooting steps.