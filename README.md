# 2026-04-22

You can browse and enjoy the notebooks in this repo without any installation.

Or you can try running the notebooks in your own environment, or in Colab, Databricks, etc.


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

```
pip intall matplotlib streamlit
```


```
streamlit run https://raw.githubusercontent.com/svorobyov/036-ib/main/06-treamlit.py
```

We can, of course, implent pure cloud-based model inferencing, using the same API on 
AWS App Runner or Google Cloud Run, but it adds nothing to functionality, but will 
reduce data privacy and security.

Besides, a cloud solution will will require a substantial secutrity/authentication 
burden and attack surface.

Our solution is superior in terms of both security and functionality, provided that 
the user has access to the API code and has enough computational resources.


