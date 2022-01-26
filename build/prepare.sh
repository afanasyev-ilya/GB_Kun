wget http://konect.cc/files/download.tsv.dimacs9-E.tar.bz2
wget http://konect.cc/files/download.tsv.dimacs9-CTR.tar.bz2
wget http://konect.cc/files/download.tsv.wikipedia_link_sv.tar.bz2
wget http://konect.cc/files/download.tsv.wikipedia_link_ru.tar.bz2
wget http://konect.cc/files/download.tsv.zhishi-all.tar.bz2
wget http://konect.cc/files/download.tsv.youtube-u-growth.tar.bz2
wget http://konect.cc/files/download.tsv.orkut-links.tar.bz2
wget http://konect.cc/files/download.tsv.soc-LiveJournal1.tar.bz2
wget http://konect.cc/files/download.tsv.soc-pokec-relationships.tar.bz2
wget http://konect.cc/files/download.tsv.flickr-links.tar.bz2

for f in *.tar.bz2; do tar -xvf "$f"; done

rm download*

#for d in */ ; do old_path=$d"out"*; echo $old_path; python3 ../../gen_mtx_graph.py $old_path ./../ ; done

NAME=$(echo "${file}" | cut -d '.' -f1)

for d in */ ; do old_path=$d"out"*; echo $old_path; new_path=$(echo "$old_path" | cut -f1 -d"/"); echo "${new_path}"; python3 ../../gen_mtx_graph.py $old_path ./../${new_path}.mtx; done