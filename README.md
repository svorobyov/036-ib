# 2026-04-22, -24

## Three Models for Three Time Series 

1. You can browse and enjoy the notebooks (`03-ib.ipynb`) in this repo without any 
   additional installations.

2. Or you can try running the notebooks in your own environment, or in Colab, Databricks, etc.
   Just make sure you install the required packages in `requiremnents.txt`.


## The API Part of the Assignment

Given time constraints we opted for the simplest but securest solution:

1. The API code is stored on GitHub.

2. The user has data to analyze, but he does not send it to the API.

3. Instead, the user fetches the API code from GitHub and runs it locally in the browser 
   on his data, without ever sending the data to the API.

4. Even though the resource burden is on the user, it is completely secure since the 
   data is never sent to the third party.

5. Production-scale model inferencing should be done with the appropriate secured and 
   trusted API/cloud services


### Installation

```bash
pip install pandas lightgbm matplotlib streamlit IPython scikit-learn graphviz
```
(or `pip install -r requirements2.txt -U`)


### Running the API

```
streamlit run https://raw.githubusercontent.com/svorobyov/036-ib/main/06-treamlit.py
```

It will ask you to upload the a `.csv` data file, similar to `'data-6-.csv'` with 
two columns, of which only `date` and `location_A` will be taken into account, for 
simplicity.


### Remarks

1. We can, of course, implement pure cloud-based model inferencing, using the same API on 
AWS App Runner or Google Cloud Run, but it adds nothing to functionality but will 
reduce data privacy and security.

2. Besides, a cloud solution will will require a substantial secutrity/authentication 
burden and attack surface.

3. Our solution is superior in terms of both security and functionality, provided that 
the user has access to the API code and has enough computational resources.


