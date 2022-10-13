from .utils import (
    is_flax_available,
    is_inflect_available,
    is_onnx_available,
    is_scipy_available,
    is_torch_available,
    is_transformers_available,
    is_unidecode_available,
)


__version__ = "0.5.0.dev0"

from .configuration_utils import ConfigMixin
from .onnx_utils import OnnxRuntimeModel
from .utils import logging


if is_torch_available():
    from .modeling_utils import ModelMixin
    from .models import AutoencoderKL, UNet2DConditionModel, UNet2DModel, VQModel
    from .optimization import (
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
        get_scheduler,
    )
    from .pipeline_utils import DiffusionPipeline
    from .pipelines import DDIMPipeline, DDPMPipeline, KarrasVePipeline, LDMPipeline, PNDMPipeline, ScoreSdeVePipeline
    from .schedulers import (
        DDIMScheduler,
        DDPMScheduler,
        KarrasVeScheduler,
        PNDMScheduler,
        SchedulerMixin,
        ScoreSdeVeScheduler,
    )
    from .training_utils import EMAModel
else:
    from .utils.dummy_pt_objects import *  # noqa F403

if is_torch_available() and is_scipy_available():
    from .schedulers import LMSDiscreteScheduler
else:
    from .utils.dummy_torch_and_scipy_objects import *  # noqa F403

if is_torch_available() and is_transformers_available():
    from .pipelines import (
        LDMTextToImagePipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionPipeline,
    )
else:
    from .utils.dummy_torch_and_transformers_objects import *  # noqa F403

if is_torch_available() and is_transformers_available() and is_onnx_available():
    from .pipelines import StableDiffusionOnnxPipeline
else:
    from .utils.dummy_torch_and_transformers_and_onnx_objects import *  # noqa F403

if is_flax_available():
    from .modeling_flax_utils import FlaxModelMixin
    from .models.unet_2d_condition_flax import FlaxUNet2DConditionModel
    from .models.vae_flax import FlaxAutoencoderKL
    from .pipeline_flax_utils import FlaxDiffusionPipeline
    from .schedulers import (
        FlaxDDIMScheduler,
        FlaxDDPMScheduler,
        FlaxKarrasVeScheduler,
        FlaxLMSDiscreteScheduler,
        FlaxPNDMScheduler,
        FlaxSchedulerMixin,
        FlaxScoreSdeVeScheduler,
    )
else:
    from .utils.dummy_flax_objects import *  # noqa F403

if is_flax_available() and is_transformers_available():
    from .pipelines import FlaxStableDiffusionPipeline
else:
    from .utils.dummy_flax_and_transformers_objects import *  # noqa F403

#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import logging
import os
import sys

from . import backend, compiler, frontend, testing, utils
from ._libinfo import __version__  # noqa

if not (sys.version_info[0] >= 3 and sys.version_info[1] >= 7):
    PY3STATEMENT = "The minimal Python requirement is Python 3.7"
    raise Exception(PY3STATEMENT)

__all__ = ["backend", "compiler", "frontend", "testing", "utils"]

root_logger = logging.getLogger(__name__)
info_handle = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s <%(name)s> %(message)s")
info_handle.setFormatter(formatter)
root_logger.addHandler(info_handle)
root_logger.propagate = False

DEFAULT_LOGLEVEL = logging.getLogger().level
log_level_str = os.environ.get("LOGLEVEL", None)
LOG_LEVEL = (
    getattr(logging, log_level_str.upper())
    if log_level_str is not None
    else DEFAULT_LOGLEVEL
)
root_logger.setLevel(LOG_LEVEL)

