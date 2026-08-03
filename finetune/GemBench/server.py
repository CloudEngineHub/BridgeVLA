'''
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Adapted from https://github.com/vlc-robot/robot-3dlotus/blob/main/challenges/server.py

'''
import argparse
import msgpack_numpy
msgpack_numpy.patch()

import sys
import traceback
from flask import Flask, request, jsonify

from actioner import MyActioner
from bridgevla.utils.memory_switches import parse_bool_flag

def main(args):
    app = Flask(__name__)
    actioner = MyActioner(
        args.base_path,
        args.model_epoch,
        expect_temporal_memory=parse_bool_flag(
            args.temporal_memory, "--temporal_memory"),
        expect_spatial_memory=parse_bool_flag(
            args.spatial_memory, "--spatial_memory"),
    )

    @app.route('/memory_config', methods=['GET'])
    def memory_config():
        '''Memory ablation switches this server was launched with (validated
        against the loaded model at startup). The client fetches this once at
        startup and aborts if its own switches disagree.'''
        return jsonify(actioner.memory_config())

    @app.route('/predict', methods=['POST'])
    def predict():
        '''
        batch is a dict containing:
            instruction: str
            obs_state_dict: observations from genrobo3d.rlbench.environments.RLBenchEnv 
        '''
        data = request.get_data()
        batch = msgpack_numpy.unpackb(data, raw=False)

        try:
            action = actioner.predict(**batch)
        except Exception:
            # Return the real traceback as text/plain 500 so the client can
            # surface server-side errors (e.g. CUDA OOM) instead of a bare 500
            # HTML page. Matches finetune/memoryBench/server.py.
            tb = traceback.format_exc()
            print(tb, file=sys.stderr, flush=True)
            return tb, 500, {'Content-Type': 'text/plain'}
        action = msgpack_numpy.packb(action)
        return action

    @app.route('/finalize', methods=['POST'])
    def finalize():
        '''Episode-end hook: stitch the just-finished episode's per-step
        overlay tri-views into grid_mvt1.png / grid_mvt2.png. The client POSTs
        here after each episode's step loop (overlay viz only). Body is an
        optional msgpack dict (currently unused — the agent tracks the episode
        dir + step labels internally).'''
        data = request.get_data()
        batch = msgpack_numpy.unpackb(data, raw=False) if data else {}
        result = actioner.finalize_episode(**batch)
        return msgpack_numpy.packb(result)

    app.run(host=args.ip, port=args.port, debug=args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Actioner server')
    parser.add_argument('--ip', type=str, default="localhost")
    parser.add_argument('--port', type=int, default=13003)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--base_path', type=str, default="/PATH_TO_MODEL_FOLDER")
    parser.add_argument('--model_epoch', type=int, default=40)
    # Memory ablation switches (train/eval alignment): must match the loaded
    # model's mvt_cfg.yaml or MyActioner raises at startup. Set via
    # run_server.sh TEMPORAL_MEMORY / SPATIAL_MEMORY.
    parser.add_argument('--temporal_memory', type=str, default="true",
                        help='true/false; stage-1 temporal memory (memory 1) the model is expected to have')
    parser.add_argument('--spatial_memory', type=str, default="true",
                        help='true/false; stage-2 spatial anchor memory (memory 2) the model is expected to have')
    args = parser.parse_args()
    main(args)
