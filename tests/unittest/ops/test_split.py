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
import unittest

import numpy as np
import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor


class SplitTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SplitTestCase, self).__init__(*args, **kwargs)

    def _run_split(
        self, *, input_shape, split_size_or_sections, dim=None, input_type="float16"
    ):
        logging.info(
            f"Test input shape {input_shape}, "
            f"split_size_or_sections={split_size_or_sections}, dim={dim}"
        )

        split_op = ops.split()
        # generate torch reference result
        X_pt = get_random_torch_tensor(input_shape, input_type)
        Ys_pt = (
            torch.split(X_pt, split_size_or_sections)
            if dim is None
            else torch.split(X_pt, split_size_or_sections, dim)
        )
        target = detect_target()
        X = Tensor(shape=input_shape, dtype=input_type, name="input_0", is_input=True)
        Ys = (
            split_op(X, split_size_or_sections)
            if dim is None
            else split_op(X, split_size_or_sections, dim)
        )
        np.testing.assert_equal(len(Ys_pt), len(Ys))

        y_shapes = []
        for idx, Y in enumerate(Ys):
            Y._attrs["name"] = f"output_{idx}"
            Y._attrs["is_output"] = True
            y_shape = [d._attrs["values"][0] for d in Y._attrs["shape"]]
            logging.info(f"AITemplate output_{idx} shape: {y_shape}")
            y_shapes.append(y_shape)

        module = compile_model(Ys, target, "./tmp", "split")

        outputs = {
            f"output_{idx}": torch.empty(y_shape).cuda().half()
            for idx, y_shape in enumerate(y_shapes)
        }
        module.run_with_tensors([X_pt], outputs)

        for idx, y_pt in enumerate(Ys_pt):
            y = outputs[f"output_{idx}"]
            self.assertTrue(torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2))

    def _run_batch_split(
        self,
        *,
        batch_sizes,
        input_shape,
        split_size_or_sections,
        dim=None,
        input_type="float16",
    ):
        logging.info(
            f"Batch test: batch_sizes {batch_sizes}, input shape {input_shape}, "
            f"split_size_or_sections={split_size_or_sections}, dim={dim}"
        )

        split_op = ops.split()

        target = detect_target()
        X = Tensor(
            shape=[IntVar(values=batch_sizes, name="input_batch_0"), *input_shape],
            dtype=input_type,
            name="input_0",
            is_input=True,
        )
        Ys = (
            split_op(X, split_size_or_sections)
            if dim is None
            else split_op(X, split_size_or_sections, dim)
        )

        for idx, Y in enumerate(Ys):
            Y._attrs["name"] = f"output_{idx}"
            Y._attrs["is_output"] = True

        module = compile_model(Ys, target, "./tmp", "split")

        for batch in batch_sizes:
            logging.info(f"checking batch: {batch}")

            # generate torch reference result
            X_pt = get_random_torch_tensor([batch, *input_shape], input_type)
            Ys_pt = (
                torch.split(X_pt, split_size_or_sections)
                if dim is None
                else torch.split(X_pt, split_size_or_sections, dim)
            )

            np.testing.assert_equal(len(Ys_pt), len(Ys))

            y_shapes = [Y_pt.size() for Y_pt in Ys_pt]
            outputs = {
                f"output_{idx}": torch.empty(y_shape).cuda().half()
                for idx, y_shape in enumerate(y_shapes)
            }
            module.run_with_tensors(
                [X_pt],
                outputs,
            )

            for idx, y_pt in enumerate(Ys_pt):
                y = outputs[f"output_{idx}"]
                self.assertTrue(torch.allclose(y_pt, y, atol=1e-2, rtol=1e-2))

    def test_split(self):
        self._run_split(input_shape=[1], split_size_or_sections=1, dim=0)
        self._run_split(input_shape=[2, 1], split_size_or_sections=1, dim=0)
        self._run_split(input_shape=[2, 3], split_size_or_sections=2, dim=1)
        self._run_split(input_shape=[2, 3, 4], split_size_or_sections=10, dim=1)
        self._run_split(input_shape=[2, 3, 4], split_size_or_sections=4, dim=2)
        self._run_split(input_shape=[8, 6, 4], split_size_or_sections=2, dim=0)
        self._run_split(input_shape=[8, 6, 4], split_size_or_sections=3, dim=0)
        self._run_split(input_shape=[4097, 128, 64], split_size_or_sections=1024, dim=0)
        self._run_split(input_shape=[4097, 128, 64], split_size_or_sections=32, dim=1)

        self._run_split(input_shape=[1], split_size_or_sections=[1], dim=0)
        self._run_split(input_shape=[8, 6, 4], split_size_or_sections=[2, 3, 3], dim=0)
        self._run_split(input_shape=[8, 6, 4], split_size_or_sections=(5, 1), dim=1)
        self._run_split(input_shape=[8, 6, 4], split_size_or_sections=(2, 2), dim=2)

        # some special cases
        self._run_split(input_shape=[2, 0, 4], split_size_or_sections=4, dim=-2)
        self._run_split(input_shape=[2, 0, 4], split_size_or_sections=0, dim=-2)
        self._run_split(input_shape=[2, 0, 4], split_size_or_sections=2, dim=-1)
        self._run_split(input_shape=[2, 0, 7], split_size_or_sections=[2, 3, 2], dim=-1)

    def test_batch_split(self):
        self._run_batch_split(
            batch_sizes=[1, 1], input_shape=[2, 1], split_size_or_sections=1, dim=1
        )
        self._run_batch_split(
            batch_sizes=[3, 4], input_shape=[2, 3, 4], split_size_or_sections=2, dim=2
        )
        self._run_batch_split(
            batch_sizes=[3, 4], input_shape=[2, 3, 4], split_size_or_sections=2, dim=3
        )
        self._run_batch_split(
            batch_sizes=[11, 5, 9],
            input_shape=[2, 9, 4],
            split_size_or_sections=[2, 4, 3],
            dim=2,
        )

        self._run_batch_split(
            batch_sizes=[11, 5, 9],
            input_shape=[4, 0, 4],
            split_size_or_sections=2,
            dim=1,
        )
        self._run_batch_split(
            batch_sizes=[11, 5, 9],
            input_shape=[4, 0, 5],
            split_size_or_sections=[1, 2, 2],
            dim=3,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()