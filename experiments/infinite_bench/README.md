### InfiniteBench

[InfiniteBench](https://github.com/OpenBMB/InfiniteBench)  evaluates language models on super long context processing through various tasks: `kv_retrieval`, `longbook_choice_eng`, `math_find`, `longbook_qa_chn`, `longbook_qa_eng`, `longdialogue_qa_eng`, `code_debug`, `longbook_sum_eng`, `number_string`, `passkey`.

1. Install requirement:

```bash
pip install -r requirement.txt
```

2. Run InfiniteBench with `SnapKV`:

```bash
bash run_infinitebench_snap.sh 
```

3. You can add custom running script. The script format is same as Longbench.