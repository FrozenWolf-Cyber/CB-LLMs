import concepts

example_name = {'SetFit/sst2': 'text', 'ag_news': 'text', 'yelp_polarity': 'text', 'dbpedia_14': 'content'}
concepts_from_labels = {'SetFit/sst2': ["negative","positive"], 'yelp_polarity': ["negative","positive"], 'ag_news': ["world", "sports", "business", "technology"], 'dbpedia_14': ["company","education","artist","athlete","office","transportation","building","natural","village","animal","plant","album","film","written"]}
class_num = {'SetFit/sst2': 2, 'ag_news': 4, 'yelp_polarity': 2, 'dbpedia_14': 14}
epoch = {'SetFit/sst2': 3, 'ag_news': 1, 'yelp_polarity': 1, 'dbpedia_14': 1}

concept_set = {'SetFit/sst2': concepts.sst2, 'yelp_polarity': concepts.yelpp, 'ag_news': concepts.agnews, 'dbpedia_14': concepts.dbpedia}