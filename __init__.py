# Copyright 2025 zhengxyz123
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# only import if running as a custom node
try:
    import comfy.utils
except ImportError:
    pass
else:
    from .clipseg import NODE_CLASS_MAPPINGS

    __all__ = ["NODE_CLASS_MAPPINGS"]
