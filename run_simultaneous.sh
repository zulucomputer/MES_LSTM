for alpha in 0.1 0.05 0.01
    do
    for country in 'Angola' 'Botswana' 'Comoros' 'Democratic_Republic_of_Congo' 'Eswatini' 'Lesotho' 'Madagascar' 'Malawi' 'Mauritius' 'Mozambique' 'Namibia' 'South_Africa' 'Tanzania' 'Zambia' 'Zimbabwe'
        do
        jbsub -cores 1x4+0 -out logs/log_${country}_${alpha}.txt -mem 5G -queue x86_6h \
        "source activate rnn0; \
        python /dccstor/eevdata/mathonsi/tutorials/MES_LSTM/run_multi_MES_LSTM.py \
        --country ${country} \
        --alpha ${alpha}"
    done

    for country in 'Seychelles'
        do
        for thresh in 0.45
            do
            jbsub -cores 1x4+0 -out logs/log_${country}.txt -mem 5G -queue x86_6h \
            "source activate rnn0; \
            python /dccstor/eevdata/mathonsi/tutorials/MES_LSTM/run_multi_MES_LSTM.py \
            --country ${country} \
            --thresh ${thresh} \
            --alpha ${alpha}"
        done
    done
done

# source activate rnn0
# python /dccstor/eevdata/mathonsi/tutorials/MES_LSTM/run_multi_MES_LSTM.py  --country ${country} --thresh ${thresh}
# jbsub -q x86_6h -cores 1x4+0 -mem 5G -out logs/log_Seychelles.txt python /dccstor/eevdata/mathonsi/tutorials/MES_LSTM/run_multi_MES_LSTM.py  --country 'Seychelles' --thresh 0.45 --alpha 0.1