#!/bin/sh
# Wait on the Modal app itself, not on a local launch log: `modal run --detach` returns
# before the run does, and a restarted session loses the log's completion line.
#   sh wait_app.sh ap-XXXXXXXX
APP=${1:?app id}
while MODAL_PROFILE=chromatic modal app list 2>/dev/null | grep -q "$APP.*ephemeral"; do
    sleep 60
done
