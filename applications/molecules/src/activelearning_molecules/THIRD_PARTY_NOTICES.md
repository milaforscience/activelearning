# Third-party notices

## S3-GFN

The S3-GFN sampler contains code adapted from
[hyeonahkimm/s3gfn](https://github.com/hyeonahkimm/s3gfn), commit
`43aa7b310e9e03ef71ea0bd0cce501a48b6e2d52`.

Copyright (c) 2024 Seonghwan Seo
Copyright (c) 2026 Hyeonah Kim

The adaptation is limited to the active-learning integration and follows these
upstream sources:

- [`train.py`](https://github.com/hyeonahkimm/s3gfn/blob/43aa7b310e9e03ef71ea0bd0cce501a48b6e2d52/src/s3gfn/train.py):
  sequence generation, trajectory-balance training, and the contrastive
  auxiliary objective.
- [`replay_buffer.py`](https://github.com/hyeonahkimm/s3gfn/blob/43aa7b310e9e03ef71ea0bd0cce501a48b6e2d52/src/s3gfn/replay_buffer.py):
  reward-diverse and FIFO replay policies.
- [`synthesizability.py`](https://github.com/hyeonahkimm/s3gfn/blob/43aa7b310e9e03ef71ea0bd0cce501a48b6e2d52/src/s3gfn/synthesizability.py):
  strict SA-score filtering.

The local files are deliberately smaller than the upstream training
application and add canonical-SMILES validation, acquisition rewards, and a
multi-fidelity terminal action for this repository's active-learning loop.

The adapted source is available under the MIT License:

```text
MIT License

Copyright (c) 2024 Seonghwan Seo
Copyright (c) 2026 Hyeonah Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
