jbsub -cores 1x12+1 -out logs/log_results.txt -mem 15G -queue x86_24h \
"source activate rnn01; \
python /dccstor/eevdata/mathonsi/tutorials/MES_LSTM/hps/save_hp.py"