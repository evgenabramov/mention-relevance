# mention-relevance
Mention relevance crowdsourcing project

Create & activate conda environment with requirements:
```bash
conda env create -f environment.yml
conda activate smelter-env
```

Run script to make predictions:
```bash
python process.py -i <input.csv> -o <output.csv>
```

Remove conda environment:
```bash
conda remove --name smelter-env --all
```
