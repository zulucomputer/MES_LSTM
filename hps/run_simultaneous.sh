for lstm_size in 70
    do
    for epochs in 35 55
        do
        jbsub -cores 1x1+0 -out logs/log_${lstm_size}_${epochs}.txt -mem 1G -queue x86_6h \
        "source activate rnn01; \
        python /dccstor/eevdata/mathonsi/tutorials/MES_LSTM/hps/run_MES_LSTM.py \
        --lstm_size ${lstm_size} \
        --epochs ${epochs}"
    done
done

# source activate rnn01
# python /dccstor/eevdata/mathonsi/tutorials/MES_LSTM/hps/run_MES_LSTM.py  --lstm_size ${lstm_size} --epochs ${epochs}