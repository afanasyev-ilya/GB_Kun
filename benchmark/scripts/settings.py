from enum import Enum

UNDIRECTED_PREFIX = "undir_"

# "fast" - very fast mode (only small graphs),
# "medium" - medium (small-sized and medium-sized graphs)
# "full" - full (test on all available graphs)
# "one_large" - one large graph per category,
# "rating" - rating mode

print_timings = True


benchmark_args = {"bfs": [ ["-top-down"]],}


available_formats = ["CSR", "COO", "CSR_SEG", "LAV", "SELL_C", "SORT"] # TODO


def requires_undir_graphs(app_name):
    #for undir_apps in ["cc", "coloring"]:
    #    if undir_apps in app_name:
    #        return True
    return False


common_iterations = 10
perf_pattern = "AVG_PERF"
correctness_pattern = "error_count:"

#DATASETS_DIR = "./datasets/"
DATASETS_DIR = "/home/afanasyev/LAGraph/src/benchmark/mtx_graphs"
GRAPHS_DIR = DATASETS_DIR + "/input_graphs/"
SOURCE_GRAPH_DIR = DATASETS_DIR + "/source_graphs/"
MTX_GENERATOR_BIN_NAME = "gen"

PERF_DATA_FILE = "./perf_stats.txt"
SCALING_FILE = "./scaling.txt"
SCALING_STEP = 8
SCALING_FOLDER_NAME = "./scaling"
SCALING_ROW_DATA_NAME = "row_data.txt"

GENERATE_UNDIRECTED = False



# how to add new graph with new category
# 1. add link (http://konect.cc/networks/amazon/ for example) to both dicts
# 2.

all_konect_graphs_data = {
    'soc_catster': {'link': 'petster-friendships-cat'},
    'soc_libimseti': {'link': 'libimseti'},
    'soc_dogster': {'link': 'petster-friendships-dog'},
    'soc_catster_dogster': {'link': 'petster-carnivore'},
    'soc_youtube_friendships': {'link': 'com-youtube'},
    'soc_pokec': {'link': 'soc-pokec-relationships'},
    'soc_orkut': {'link': 'orkut-links'},
    'soc_livejournal': {'link': 'soc-LiveJournal1'},
    'soc_livejournal_links': {'link': 'livejournal-links'},
    'soc_twitter_www': {'link': 'twitter'},
    'soc_friendster': {'link': 'friendster'},
    'soc_flick': {'link': 'flickr-growth'},
    'soc_livemocha': {'link': 'livemocha'},



    'web_stanford': {'link': 'web-Stanford'},
    'web_baidu_internal': {'link': 'zhishi-baidu-internallink'},
    'web_wikipedia_links_fr': {'link': 'wikipedia_link_fr'},
    'web_wikipedia_links_ru': {'link': 'wikipedia_link_ru'},
    'web_zhishi': {'link': 'zhishi-all'},
    'web_wikipedia_links_en': {'link': 'wikipedia_link_en'},
    'web_dbpedia_links': {'link': 'dbpedia-link'},
    'web_uk_domain_2002': {'link': 'dimacs10-uk-2002'},
    'web_web_trackers': {'link': 'trackers-trackers', 'unarch_graph_name': 'trackers'},
    'web_wikipedia_links_it': {'link': 'wikipedia_link_it'},
    'web_wikipedia_links_sv': {'link': 'wikipedia_link_sv'},
    'web_dimacs10-cnr-2000': {'link': 'dimacs10-cnr-2000'},
    'web_eu_2005': {'link': 'dimacs10-eu-2005'},
    'web_hudong': {'link': 'zhishi-hudong-relatedpages'},


    'road_colorado': {'link': 'dimacs9-COL'},
    'road_texas': {'link': 'roadNet-TX'},
    'road_california': {'link': 'roadNet-CA'},
    'road_eastern_usa': {'link': 'dimacs9-E'},
    'road_western_usa': {'link': 'dimacs9-W'},
    'road_central_usa': {'link': 'dimacs9-CTR'},
    'road_full_usa': {'link': 'dimacs9-USA'},

    'rating_epinions': {'link': 'epinions-rating'},
    'rating_amazon_ratings': {'link': 'amazon-ratings'},
    'rating_yahoo_songs': {'link': 'yahoo-song'},

    'GAP-road': {'link': 'https://suitesparse-collection-website.herokuapp.com/MM/GAP/GAP-road.tar.gz'},
    'GAP-web': {'link': 'https://suitesparse-collection-website.herokuapp.com/MM/GAP/GAP-web.tar.gz'},
    'GAP-kron': {'link': 'https://suitesparse-collection-website.herokuapp.com/MM/GAP/GAP-kron.tar.gz'},
    'GAP-urand': {'link': 'https://suitesparse-collection-website.herokuapp.com/MM/GAP/GAP-urand.tar.gz'},

    # new graph, taken from http://konect.cc/networks/digg-friends/ , name (key) is arbitrary
    # link (value) taken as a part of "Data as TSV" -> copy_link http://konect.cc/files/download.tsv. [digg-friends] .tar.bz2
    'soc_digg': {'link': 'digg-friends'}
}

#####################

konect_tiny_only = ['road_california', 'soc_catster_dogster', 'soc_libimseti', 'soc_pokec', 'soc_flick']
syn_tiny_only = ["syn_rmat_18_32", "syn_ru_18_32", "syn_rmat_20_32", "syn_ru_20_32"]

#####################

konect_small_only = ['soc_livejournal', 'web_zhishi', 'road_full_usa', 'web_wikipedia_links_ru', 'web_wikipedia_links_it']
syn_small_only = ["syn_rmat_22_32", "syn_ru_22_32"]

#####################

konect_medium_only = ['web_wikipedia_links_sv', 'soc_orkut', 'web_web_trackers', 'web_dbpedia_links', 'web_uk_domain_2002']
syn_medium_only = ["syn_rmat_23_32", "syn_ru_23_32"]

#####################

konect_large_only = ['GAP-kron', 'GAP-urand', 'GAP-web', 'soc_twitter_www', 'soc_friendster']
syn_large_only = []

#####################

konect_tiny_small = konect_tiny_only + konect_small_only
syn_tiny_small = syn_tiny_only + syn_small_only

konect_tiny_small_medium = konect_tiny_only + konect_small_only + konect_medium_only
syn_tiny_small_medium = syn_tiny_only + syn_small_only + syn_medium_only

#####################

syn_scaling = ["syn_rmat_18_32", "syn_rmat_19_32", "syn_rmat_20_32", "syn_rmat_21_32", "syn_rmat_22_32",
               "syn_rmat_23_32", "syn_ru_18_32", "syn_ru_19_32", "syn_ru_20_32", "syn_ru_21_32", "syn_ru_22_32",
               "syn_ru_23_32"]
konect_scaling = []

#####################

# new sets of arrays, first for synthetic, second for real-wold graphs.
syn_deep_learning = []
konect_deep_learning = ['soc_digg'] # our new digg graph here (its key)

#####################

apps_and_graphs_ingore = {"sssp": [],
                          "bfs": []}

konect_fastest = ['soc_catster_dogster', 'road_texas']
syn_fastest = ["syn_rmat_18_32", "syn_ru_18_32"]

konect_best = ['soc_libimseti', 'soc_pokec', 'web_dimacs10-cnr-2000', 'web_eu_2005', 'soc_livemocha',
               'soc_livejournal', 'web_zhishi', 'web_wikipedia_links_ru', 'web_wikipedia_links_it', 'web_hudong',
               'web_wikipedia_links_sv', 'soc_orkut', 'web_web_trackers', 'web_dbpedia_links', 'web_uk_domain_2002',
               'soc_twitter_www', 'soc_friendster']

konect_best_fast = ['soc_libimseti', 'soc_pokec', 'web_dimacs10-cnr-2000', 'web_eu_2005', 'soc_livemocha',
               'soc_livejournal', 'web_zhishi', 'web_wikipedia_links_ru', 'web_wikipedia_links_it', 'web_hudong',
               'web_wikipedia_links_sv', 'soc_orkut', 'web_web_trackers']

