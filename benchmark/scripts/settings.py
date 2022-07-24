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
    'South_African_companies': {'link': 'brunson_south-africa'},
    'Corporate_club_memberships': {'link': 'brunson_club-membership'},
    'Corporate_leaderships': {'link': 'brunson_corporate-leadership'},
    'American_Revolution': {'link': 'brunson_revolution'},
    'Record_labels': {'link': 'dbpedia-recordlabel'},
    'Occupations': {'link': 'dbpedia-occupation'},
    'YouTube': {'link': 'youtube-groupmemberships'},
    'DBpedia_countries': {'link': 'dbpedia-country'},
    'Teams': {'link': 'dbpedia-team'},
    'Actor_movies': {'link': 'actor-movie'},
    'IMDB': {'link': 'actor2'},
    'Flickr': {'link': 'flickr-growth'},
    'Wikinews_edits_(sr)': {'link': 'edit-srwikinews'},
    'Wikipedia_edits_(az)': {'link': 'edit-azwiki'},
    'Wiktionary_edits_(ko)': {'link': 'edit-kowiktionary'},
    'Wiktionary_edits_(nl)': {'link': 'edit-nlwiktionary'},
    'Wikipedia_edits_(ms)': {'link': 'edit-mswiki'},
    'Wiktionary_edits_(el)': {'link': 'edit-elwiktionary'},
    'Wikipedia_edits_(gl)': {'link': 'edit-glwiki'},
    'Wikipedia_edits_(et)': {'link': 'edit-etwiki'},
    'Wikipedia_edits_(hr)': {'link': 'edit-hrwiki'},
    'Wikipedia_edits_(sl)': {'link': 'edit-slwiki'},
    'Wikinews_edits_(en)': {'link': 'edit-enwikinews'},
    'Wikipedia_edits_(lt)': {'link': 'edit-ltwiki'},
    'Wiktionary_edits_(es)': {'link': 'edit-eswiktionary'},
    'Wikipedia_edits_(simple)': {'link': 'edit-simplewiki'},
    'Wikipedia_edits_(hy)': {'link': 'edit-hywiki'},
    'Wiktionary_edits_(zh)': {'link': 'edit-zhwiktionary'},
    'Wikipedia_edits_(el)': {'link': 'edit-elwiki'},
    'Wikipedia_edits_(th)': {'link': 'edit-thwiki'},
    'Wikipedia_edits_(eu)': {'link': 'edit-euwiki'},
    'Wiktionary_edits_(pl)': {'link': 'edit-plwiktionary'},
    'Wiktionary_edits_(de)': {'link': 'edit-dewiktionary'},
    'Wikipedia_edits_(sk)': {'link': 'edit-skwiki'},
    'Wikipedia_edits_(eo)': {'link': 'edit-eowiki'},
    'Wikipedia_edits_(war)': {'link': 'edit-warwiki'},
    'Wikipedia_edits_(bg)': {'link': 'edit-bgwiki'},
    'Wikiquote_edits_(fr)': {'link': 'edit-frwikisource'},
    'Wikiquote_edits_(en)': {'link': 'edit-enwikisource'},
    'Wikipedia_edits_(da)': {'link': 'edit-dawiki'},
    'Wiktionary_edits_(ru)': {'link': 'edit-ruwiktionary'},
    'DBLP': {'link': 'dblp_coauthor'},
    'Wikipedia_edits_(ro)': {'link': 'edit-rowiki'},
    'Wikipedia_edits_(id)': {'link': 'edit-idwiki'},
    'Wikipedia_edits_(cs)': {'link': 'edit-cswiki'},
    'Wikipedia_edits_(sr)': {'link': 'edit-srwiki'},
    'Wikipedia_edits_(fi)': {'link': 'edit-fiwiki'},
    'Wikipedia_edits_(tr)': {'link': 'edit-trwiki'},
    'Wikipedia_edits_(ko)': {'link': 'edit-kowiki'},
    'Wikipedia_edits_(no)': {'link': 'edit-nowiki'},
    'Wikipedia_edits_(ceb)': {'link': 'edit-cebwiki'},
    'Wikipedia_edits_(ca)': {'link': 'edit-cawiki'},
    'Wikipedia_edits_(hu)': {'link': 'edit-huwiki'},
    'Wikipedia_edits_(he)': {'link': 'edit-hewiki'},
    'Wikipedia_edits_(fa)': {'link': 'edit-fawiki'},
    'Wikipedia_edits_(uk)': {'link': 'edit-ukwiki'},
    'Wikipedia_edits_(ar)': {'link': 'edit-arwiki'},
    'Wiktionary_edits_(mg)': {'link': 'edit-mgwiktionary'},
    'Wiktionary_edits_(fr)': {'link': 'edit-frwiktionary'},
    'Wikipedia_edits_(vi)': {'link': 'edit-viwiki'},
    'Wikipedia_edits_(pt)': {'link': 'edit-ptwiki'},
    'Wikipedia_edits_(sv)': {'link': 'edit-svwiki'},
    'Wikipedia_edits_(zh)': {'link': 'edit-zhwiki'},
    'Wikipedia_edits_(sh)': {'link': 'edit-shwiki'},
    'Wikipedia_edits_(pl)': {'link': 'edit-plwiki'},
    'Wikipedia_edits_(nl)': {'link': 'edit-nlwiki'},
    'Wikipedia_edits_(ja)': {'link': 'edit-jawiki'},
    'Wiktionary_edits_(en)': {'link': 'edit-enwiktionary'},
    'Wikipedia_edits_(ru)': {'link': 'edit-ruwiki'},
    'US_patents': {'link': 'patentcite'},
    'arXiv_hep-th': {'link': 'ca-cit-HepTh'},
    'arXiv_hep-ph': {'link': 'ca-cit-HepPh'},
    'Wikipedia_talk_(ru)': {'link': 'wiki_talk_ru'},
    'Wikipedia_talk_(pt)': {'link': 'wiki_talk_pt'},
    'Wikipedia_threads_(de)': {'link': 'wikipedia-discussions-de'},
    'Wikipedia_talk_(zh)': {'link': 'wiki_talk_zh'},
    'Wikipedia_talk_(es)': {'link': 'wiki_talk_es'},
    'Wikipedia_messages_(en)': {'link': 'wiki-Talk'},
    'Wikipedia_talk_(it)': {'link': 'wiki_talk_it'},
    'Wikipedia_talk_(fr)': {'link': 'wiki_talk_fr'},
    'Wikipedia_talk_(de)': {'link': 'wiki_talk_de'},
    'Wikipedia_talk_(en)': {'link': 'wiki_talk_en'},
    'Skitter': {'link': 'as-skitter'},
    'vi.sualize.us_tag–item': {'link': 'pics_ti'},
    'Discogs_label–genre': {'link': 'discogs_lgenre'},
    'TV_Tropes': {'link': 'dbtropes-feature'},
    'Wikipedia_categories_(en)': {'link': 'wiki-en-cat'},
    'Discogs_label–style': {'link': 'discogs_lstyle'},
    'BibSonomy_tag–item': {'link': 'bibsonomy-2ti'},
    'CiteULike_tag–item': {'link': 'citeulike-ti'},
    'Discogs_artist–genre': {'link': 'discogs_genre'},
    'Discogs_artist–style': {'link': 'discogs_style'},
    'Wikipedia_links_(af)': {'link': 'wikipedia_link_af'},
    'Wikipedia_links_(ia)': {'link': 'wikipedia_link_ia'},
    'Wikipedia_links_(ast)': {'link': 'wikipedia_link_ast'},
    'Wikipedia_links_(ml)': {'link': 'wikipedia_link_ml'},
    'Wikipedia_links_(bpy)': {'link': 'wikipedia_link_bpy'},
    'Wikipedia_links_(tl)': {'link': 'wikipedia_link_tl'},
    'Stanford': {'link': 'web-Stanford'},
    'Wikipedia_links_(sq)': {'link': 'wikipedia_link_sq'},
    'Wikipedia_links_(be-x-old)': {'link': 'wikipedia_link_be_x_old'},
    'Wikipedia_links_(ne)': {'link': 'wikipedia_link_ne'},
    'Wikipedia_links_(bn)': {'link': 'wikipedia_link_bn'},
    'Wikipedia_links_(te)': {'link': 'wikipedia_link_te'},
    'Wikipedia_links_(cy)': {'link': 'wikipedia_link_cy'},
    'Italian_CNR': {'link': 'dimacs10-cnr-2000'},
    'Baidu_related': {'link': 'zhishi-baidu-relatedpages'},
    'Wikipedia_links_(lmo)': {'link': 'wikipedia_link_lmo'},
    'Wikipedia_links_(lv)': {'link': 'wikipedia_link_lv'},
    'Wikipedia_links_(nn)': {'link': 'wikipedia_link_nn'},
    'Wikipedia_links_(la)': {'link': 'wikipedia_link_la'},
    'Wikipedia_links_(sa)': {'link': 'wikipedia_link_sa'},
    'Wikipedia_links_(uz)': {'link': 'wikipedia_link_uz'},
    'Wikipedia_links_(az)': {'link': 'wikipedia_link_az'},
    'Wikipedia_links_(zh-min-nan)': {'link': 'wikipedia_link_zh_min_nan'},
    'Wikipedia_links_(tt)': {'link': 'wikipedia_link_tt'},
    'Wikipedia_links_(mk)': {'link': 'wikipedia_link_mk'},
    'Wikipedia_links_(ka)': {'link': 'wikipedia_link_ka'},
    'Wikipedia_links_(simple)': {'link': 'wikipedia_link_simple'},
    'Wikipedia_links_(et)': {'link': 'wikipedia_link_et'},
    'Wikipedia_links_(bug)': {'link': 'wikipedia_link_bug'},
    'Wikipedia_links_(tg)': {'link': 'wikipedia_link_tg'},
    'Google_hyperlinks': {'link': 'web-Google'},
    'Wikipedia_links_(el)': {'link': 'wikipedia_link_el'},
    'Wikipedia_links_(th)': {'link': 'wikipedia_link_th'},
    'Wikipedia_links_(lt)': {'link': 'wikipedia_link_lt'},
    'Wikipedia_links_(be)': {'link': 'wikipedia_link_be'},
    'Wikipedia_links_(gl)': {'link': 'wikipedia_link_gl'},
    'Wikipedia_links_(ur)': {'link': 'wikipedia_link_ur'},
    'Wikipedia_links_(ce)': {'link': 'wikipedia_link_ce'},
    'Wikipedia_links_(hr)': {'link': 'wikipedia_link_hr'},
    'Wikipedia_links_(sk)': {'link': 'wikipedia_link_sk'},
    'Wikipedia_links_(ta)': {'link': 'wikipedia_link_ta'},
    'Wikipedia_links_(hi)': {'link': 'wikipedia_link_hi'},
    'Berkeley_Stanford': {'link': 'web-BerkStan'},
    'Wikipedia_links_(sl)': {'link': 'wikipedia_link_sl'},
    'Wikipedia_links_(eu)': {'link': 'wikipedia_link_eu'},
    'TREC_WT10g': {'link': 'trec-wt10g'},
    'Wikipedia_links_(bg)': {'link': 'wikipedia_link_bg'},
    'Wikipedia_links_(da)': {'link': 'wikipedia_link_da'},
    'Wikipedia_links_(eo)': {'link': 'wikipedia_link_eo'},
    'Wikipedia_links_(bs)': {'link': 'wikipedia_link_bs'},
    'Wikipedia_links_(kk)': {'link': 'wikipedia_link_kk'},
    'Wikipedia_links_(he)': {'link': 'wikipedia_link_he'},
    'Wikipedia_links_(tr)': {'link': 'wikipedia_link_tr'},
    'Wikipedia_links_(fi)': {'link': 'wikipedia_link_fi'},
    'Indian_domain': {'link': 'dimacs10-in-2004'},
    'Hudong_internal': {'link': 'zhishi-hudong-internallink'},
    'European_Union_domain': {'link': 'dimacs10-eu-2005'},
    'Baidu_internal': {'link': 'zhishi-baidu-internallink'},
    'Wikipedia_links_(cs)': {'link': 'wikipedia_link_cs'},
    'Wikipedia_links_(hy)': {'link': 'wikipedia_link_hy'},
    'Wikipedia_links_(no)': {'link': 'wikipedia_link_no'},
    'Wikipedia_links_(oc)': {'link': 'wikipedia_link_oc'},
    'Hudong_related': {'link': 'zhishi-hudong-relatedpages'},
    'Wikipedia_links_(ms)': {'link': 'wikipedia_link_ms'},
    'Wikipedia_links_(ko)': {'link': 'wikipedia_link_ko'},
    'Wikipedia_links_(ro)': {'link': 'wikipedia_link_ro'},
    'Wikipedia_links_(id)': {'link': 'wikipedia_link_id'},
    'Wikipedia_links_(war)': {'link': 'wikipedia_link_war'},
    'Wikipedia_links_(ca)': {'link': 'wikipedia_link_ca'},
    'Wikipedia_dynamic_(nl)': {'link': 'link-dynamic-nlwiki'},
    'Wikipedia_links_(hu)': {'link': 'wikipedia_link_hu'},
    'Wikipedia_links_(es)': {'link': 'wikipedia_link_es'},
    'Wikipedia_dynamic_(pl)': {'link': 'link-dynamic-plwiki'},
    'Wikipedia_links_(vi)': {'link': 'wikipedia_link_vi'},
    'Wikipedia_links_(pt)': {'link': 'wikipedia_link_pt'},
    'Wikipedia_links_(uk)': {'link': 'wikipedia_link_uk'},
    'Wikipedia_links_(nl)': {'link': 'wikipedia_link_nl'},
    'Wikipedia_dynamic_(it)': {'link': 'link-dynamic-itwiki'},
    'Zhishi': {'link': 'zhishi-all'},
    'Wikipedia_growth_(en)': {'link': 'wikipedia-growth'},
    'Wikipedia_dynamic_(fr)': {'link': 'link-dynamic-frwiki'},
    'Florida': {'link': 'dimacs9-FLA'},
    'Northwest_USA': {'link': 'dimacs9-NW'},
    'California': {'link': 'roadNet-CA'},
    'Northeast_USA': {'link': 'dimacs9-NE'},
    'California_and_Nevada': {'link': 'dimacs9-CAL'},
    'Great_Lakes': {'link': 'dimacs9-LKS'},
    'Eastern_USA': {'link': 'dimacs9-E'},
    'Western_USA': {'link': 'dimacs9-W'},
    'Central_USA': {'link': 'dimacs9-CTR'},
    'Full_USA': {'link': 'dimacs9-USA'},
    'vi.sualize.us_user–tag': {'link': 'pics_ut'},
    'vi.sualize.us_user–item': {'link': 'pics_ui'},
    'BibSonomy_user–tag': {'link': 'bibsonomy-2ut'},
    'BibSonomy_user–item': {'link': 'bibsonomy-2ui'},
    'CiteULike_user–tag': {'link': 'citeulike-ut'},
    'CiteULike_user–item': {'link': 'citeulike-ui'},
    'Prosper_loans': {'link': 'prosper-loans'},
    'Twitter_user–tag': {'link': 'munmun_twitterex_ut'},
    'Last.fm_bands': {'link': 'lastfm_band'},
    'Last.fm_songs': {'link': 'lastfm_song'},
    'Yahoo_advertisers': {'link': 'lasagne-yahoo'},
    'Amazon_(TWEB,_0601)': {'link': 'amazon0601'},
    'DBpedia': {'link': 'dbpedia-all'},
    'Actor_collaborations': {'link': 'actor-collaboration'},
    'Wikipedia_conflict': {'link': 'wikiconflict'},
    'Stack_Overflow': {'link': 'sx-stackoverflow'},
    'Livemocha': {'link': 'livemocha'},
    'Dogster_households': {'link': 'petster-dog-household'},
    'Catster_dogster_households': {'link': 'petster-catdog-household'},
    'Youtube_friendships': {'link': 'com-youtube'},
    'Hyves': {'link': 'hyves'},
    'Catster': {'link': 'petster-friendships-cat'},
    'Youtube_links': {'link': 'youtube-links'},
    'Catster_friends': {'link': 'petster-cat-friend'},
    'Dogster': {'link': 'petster-friendships-dog'},
    'Flixster': {'link': 'flixster'},
    'Dogster_friends': {'link': 'petster-dog-friend'},
    'Higgs': {'link': 'higgs-twitter-social'},
    'Catster_dogster_friends': {'link': 'petster-catdog-friend'},
    'Flickr_links': {'link': 'flickr-links'},
    'Catster_Dogster': {'link': 'petster-carnivore'},
    'YouTube': {'link': 'youtube-u-growth'},
    'Libimseti.cz': {'link': 'libimseti'},
    'Pokec': {'link': 'soc-pokec-relationships'},
    'LiveJournal_links': {'link': 'livejournal-links'},
    'Jester_100': {'link': 'jester1'},
    'Digg_votes': {'link': 'digg-votes'},
    'Amazon_ratings': {'link': 'amazon-ratings'},
    'MovieLens_10M': {'link': 'movielens-10m_rating'},
    'Epinions': {'link': 'epinions-rating'},
    'Wikipedia_words_(en)': {'link': 'gottron-excellent'},
    'Enron_words': {'link': 'bag-enron'},
    'WebUni_Magdeburg': {'link': 'webuni'}
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
konect_deep_learning = ['IMDB', 'Flickr', 'Discogs', 'Wiktionary_edits_(ku)', 'Wiktionary_edits_(hu)', 'Wiktionary_edits_(pt)', 'Wikipedia_edits_(ta)', 'Wikivoyage_edits_(en)', 'Wikipedia_edits_(bs)', 'Wikipedia_edits_(kk)', 'Wikibooks_edits_(en)', 'Wikipedia_edits_(ml)', 'Wikipedia_edits_(be)', 'Wiktionary_edits_(io)', 'Wikipedia_edits_(lv)', 'Wikipedia_edits_(bn)', 'Wikiquote_edits_(ru)', 'Wikipedia_edits_(ur)', 'Wikipedia_edits_(mk)', 'Wiktionary_edits_(sv)', 'Wikipedia_edits_(cy)', 'Wikipedia_edits_(nn)', 'Wikipedia_edits_(la)', 'Wikiquote_edits_(de)', 'Wikipedia_edits_(hi)', 'Wiktionary_edits_(it)', 'Wiktionary_edits_(fi)', 'Wikipedia_edits_(vo)', 'Wikipedia_edits_(ka)']

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

konect_twenty_set = ['soc_libimseti', 'soc_pokec', 'web_dimacs10-cnr-2000', 'web_eu_2005', 'soc_livemocha',
                     'soc_livejournal', 'web_zhishi', 'web_wikipedia_links_ru', 'web_wikipedia_links_it', 'web_hudong',
                     'web_wikipedia_links_sv', 'soc_orkut', 'web_web_trackers', 'web_dbpedia_links', 'web_uk_domain_2002',
                     'soc_twitter_www', 'soc_friendster', 'soc_catster_dogster', 'road_texas', 'road_colorado']
