for lstm_size in 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150
    do
    for epochs in 15 20 25 30 35 40 45 50 55 60 65 70 75
        do
        for batch_size in 8 16 32 64
            do
            for window in 7 14
                do
                jbsub -cores 1x1+0 -out logs/log_${lstm_size}_${epochs}_${batch_size}_${window}.txt -mem 1G -queue x86_6h \
                "source activate rnn01; \
                python /dccstor/eevdata/mathonsi/tutorials/MES_LSTM/hps/save_hp.py \
                --lstm_size ${lstm_size} \
                --epochs ${epochs} \
                --batch_size ${batch_size} \
                --window ${window}"
            done
        done
    done
done