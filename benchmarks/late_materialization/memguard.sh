#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Run a command with a resident memory and a wall-clock limit.
#
# Usage: memguard.sh <max_rss_mb> <timeout_s> command [args...]
#
# macOS does not enforce `ulimit -v`, so this polls the RSS of the command
# every 0.2 s and kills it with SIGKILL when it exceeds max_rss_mb. A very fast
# allocation can overshoot the limit between two polls. Only the command's own
# process is measured, not its children.
#
# Exit status: the command's status, 137 if killed for memory, 124 if killed
# for time.

set -u

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <max_rss_mb> <timeout_s> command [args...]" >&2
  exit 2
fi

max_kb=$(( $1 * 1024 ))
timeout=$2
shift 2

# A background job in a non-interactive shell reads stdin from /dev/null;
# pass ours through explicitly.
exec 3<&0
"$@" <&3 &
pid=$!
start=$SECONDS

while kill -0 "$pid" 2>/dev/null; do
  rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
  if [ -n "$rss" ] && [ "$rss" -gt "$max_kb" ]; then
    kill -9 "$pid"
    echo "memguard: killed $pid, RSS ${rss}KB > ${max_kb}KB" >&2
    wait "$pid"
    exit 137
  fi
  if [ $(( SECONDS - start )) -ge "$timeout" ]; then
    kill -9 "$pid"
    echo "memguard: killed $pid after ${timeout}s" >&2
    wait "$pid"
    exit 124
  fi
  sleep 0.2
done

wait "$pid"
