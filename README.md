<div style="display: flex; margin-right: -200px;align-items: center;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="KV-Compress.svg">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-light.png" style="max-width: 70%; width: auto;">
  </picture>

</div>
<h3 align="center">
KV cache compression on our favorite inference server
</h3>

---

This is a (messy) fork of vLLM v0.6.0 showcasing our new KV cache compression method that can achieve up to 5.18x throughput for single-instance deployments. Stay tuned for updates!

<div style="display: flex; justify-content: space-between; align-items: center;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="experiments/out-8b/throughtput_by_cr.jpg">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-light.png" style="max-width: 100%; width: auto;">
  </picture>
  <picture style="margin-right: 15px;">
    <source media="(prefers-color-scheme: dark)" srcset="experiments/out-8b/longbench_score_by_cr.jpg">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-light.png" style="max-width: 100%; width: auto;">
  </picture>
</div>


## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):
```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```
