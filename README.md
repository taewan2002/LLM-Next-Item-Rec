# Zero-Shot Next-Item Recommendation (Updating)
Code for the Paper "Zero-Shot Next-Item Recommendation using Large Pretrained Language Models"

![showcase_ps_prompting](gpt_rec_main.jpg)

## News

**04/10/2022**: We updated code for Zero-Shot NIR on ml-100k.<br/>
**08/07/2023**: We forked https://github.com/AGI-Edgerunners/LLM-Next-Item-Rec<br/>
**08/07/2023**: We updated LLM text-davinci-003 to lk-d2 of naverclova<br/>

## Quick Start
Set .env file
~~~
HYPER_CLOVA_KEY="YOUR_KEY"
HYPER_CLOVA_GATEWAY="YOUR_GATEWAY"
~~~
Command for Zero-Shot NIR on ml-100k 
~~~
python three_stage_0_NIR.py
~~~

## Citations

```bibtex
@article{wang2023zero,
  title={Zero-Shot Next-Item Recommendation using Large Pretrained Language Models},
  author={Wang, Lei and Lim, Ee-Peng},
  journal={arXiv preprint arXiv:2304.03153},
  year={2023}
}
```
