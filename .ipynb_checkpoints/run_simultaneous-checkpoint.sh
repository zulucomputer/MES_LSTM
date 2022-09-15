for alpha in 0.2 0.1 0.05 # significance levels
    do
    for country in 'Angola' 'Botswana' 'Comoros' 'Democratic_Republic_of_Congo' 'Eswatini' 'Lesotho' 'Madagascar' 'Malawi' 'Mauritius' 'Mozambique' 'Namibia' 'South_Africa' 'Tanzania' 'Zambia' 'Zimbabwe'
        do
        jbsub -cores 1x2+1 -out logs/log_${country}_${alpha}.txt -mem 4G -queue x86_6h \
        "source activate meslstm01; \
        python /dccstor/eevdata/mathonsi/tutorials/MES_LSTM/run_multi_MES_LSTM.py \
        --country ${country} \
        --alpha ${alpha}"
    done

    for country in 'Seychelles'
        do
        for thresh in 0.45
            do
            jbsub -cores 1x2+1 -out logs/log_${country}_${alpha}.txt -mem 4G -queue x86_6h \
            "source activate meslstm01; \
            python /dccstor/eevdata/mathonsi/tutorials/MES_LSTM/run_multi_MES_LSTM.py \
            --country ${country} \
            --thresh ${thresh} \
            --alpha ${alpha}"
        done
    done
done

# jbsub -cores 1x2+1 -out logs/log_Angola_0.2.txt -mem 4G -queue x86_24h python /dccstor/eevdata/mathonsi/tutorials/MES_LSTM/run_multi_MES_LSTM.py --country 'Angola' --alpha 0.2