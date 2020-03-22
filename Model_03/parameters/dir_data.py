######################################################################
# Representative results #############################################
######################################################################

# Loding from pre-calculated data
#dir_data='20200229_214730' # n_cells_lstm 2,4,..48 random deletion after loading '20200222_002120/20200222_122717'
#dir_data='20200311_234904' # n_cells_lstm 2,4,..48 random deletion after loading '20200222_002120/20200222_122717'
#dir_data='20200311_235109' # n_cells_lstm 2,4,..48 random deletion after loading '20200222_002120/20200222_122717'
#dir_data='20200311_235702' # n_cells_lstm 2,4,..48 random deletion after loading '20200222_002120/20200222_122717'

#dir_data='20200315_112129' # n_cells_lstm 2,4,..24 random deletion after loading 20200314_202346'
#dir_data='20200315_112201' # n_cells_lstm 2,4,..24 random deletion after loading 20200314_202346'
#dir_data='20200315_112233' # n_cells_lstm 2,4,..24 random deletion after loading 20200314_202346'

#dir_data='20200227_151151' # learning_rate 0.0001-0.1000 after loading (combined '20200222_233321', '20200223_235457' and '20200224_234232')
#dir_data='20200318_170259' # learning_rate 0.0001-0.1000 after loading '20200314_202346'

#dir_data='20200315_112233' # n_cells_lstm 2,4,..24 random deletion after loading 20200314_202346'

# Raw simulation to compare with pre-calculated
#dir_data='20200304_080520' # n_cells_lstm 5-200 (combined '20200218_212228' and '20200303_183303')
#dir_data='20200303_174857' # learning_rate 0.0001-0.1000 (combined '20200219_223846', '20200220_230830' and '')

#dir_data='20200222_002120/20200222_122717' # n_cells_lstm 48, long run (200000, stable)
#dir_data='20200314_202346' # n_cells_lstm 24, long run (200000, stable)

#dir_data='20200319_170330' # n_cells_lstm 12,14,..24 long run
#dir_data='20200321_184631' # n_cells_lstm 12,14,..24 long run
#dir_data='20200319_171859' # n_cells_lstm 12,14 long run
#dir_data='20200319_171942' # n_cells_lstm 12,14 long run
#dir_data='20200319_172028' # n_cells_lstm 12,14 long run
#dir_data='20200321_205704' # n_cells_lstm 16,18,20 long run
#dir_data='20200321_205731' # n_cells_lstm 16,18,20 long run
#dir_data='20200321_205833' # n_cells_lstm 16,18,20 long run 
#dir_data='20200322_160733' # n_cells_lstm 16,22 long run
#dir_data='20200322_160820' # n_cells_lstm 16,22 long run
#dir_data='20200321_014712' # n_cells_lstm 22,24 long run
#dir_data='20200322_160859' # n_cells_lstm 22,24 long run
#dir_data='20200322_160932' # n_cells_lstm 22,24 long run

# List of runs with stable results for each n_cells_lstm
#12 20200319_171859/20200319_171859 20200319_171942/20200319_171942 20200319_172028/20200319_172028
#14 20200319_171859/20200319_225701 20200319_171942/20200319_225727
#16 20200321_205833/20200321_205833
#18 20200319_170330/20200319_005039 20200321_184631/20200321_014642 20200321_205731/20200322_024611
#20 20200319_170330/20200319_061946 20200321_205704/20200322_081548 20200321_205731/20200322_082151
#22
#24 20200319_170330/20200319_061919


######################################################################
# All results ########################################################
######################################################################

#dir_data='20200216_191229'
#dir_data='20200216_204436'
#dir_data='20200216_233234' # n_cells_lstm 4, 15, ... 48
#dir_data='20200217_103834'

#dir_data='20200219_223846' # learning_rate 0.0001, 0.0002, ... 0.0019
#dir_data='20200220_230830' # learning_rate 0.0020, 0.0025, ... 0.0100
#dir_data='20200302_062250' # learning_rate 0.0150, 0.0200, ... 0.1000
#dir_data='20200223_153711' # combined '20200219_223846' and '20200220_230830'
#dir_data='20200222_002120' # three long runs (200000)

#dir_data='20200222_233321' # learning_rate 0.0001, 0.0002, ... 0.0019 after loading '20200222_002120/20200222_122717'
#dir_data='20200223_235457' # learning_rate 0.0020, 0.0025, ... 0.0100 after loading '20200222_002120/20200222_122717'
#dir_data='20200224_234232' # learning_rate 0.0150, 0.0200, ... 0.1000 after loading '20200222_002120/20200222_122717'
#dir_data='20200224_220741' # combined '20200222_233321' and '20200223_235457'

#dir_data='20200227_160031' # n_cells_lstm 1,2,..11 after loading '20200222_002120/20200222_122717'
#dir_data='20200227_123416' # n_cells_lstm 4,8 after loading '20200222_002120/20200222_122717'
#dir_data='20200226_200910' # n_cells_lstm 12,16,...60 after loading '20200222_002120/20200222_122717'
#dir_data='20200228_130159' # n_cells_lstm 13,14,..36 after loading '20200222_002120/20200222_122717'
#dir_data='20200226_153138' # n_cells_lstm 36,48,60 after loading '20200222_002120/20200222_122717'
#dir_data='20200227_150929' # combined '20200227_123416' and '20200226_200910' (n_cells_lstm 4-60)
#dir_data='20200228_123122' # combined '20200227_160031' and '20200226_200910' (n_cells_lstm 1-60)
#dir_data='20200229_210037' # combined '20200227_160031' and '20200228_130159' n_cells_lstm 1,2,..36 after loading '20200222_002120/20200222_122717'

#dir_data='20200218_212228' # n_cells_lstm 5, 10, ... 100
#dir_data='20200303_183303' # n_cells_lstm 110,120, ... 200

#dir_data='20200314_202238' # n_cells_lstm 24, long run (unstable)
#dir_data='20200314_202309' # n_cells_lstm 24, long run (unstable)
#dir_data='20200314_202346' # n_cells_lstm 24, long run (stable)

#dir_data='20200229_214730/20200301_012501' # n_cells_lstm=8

#dir_data='20200229_003524' # single run


######################################################################
# Parameters for combining data ######################################
######################################################################

#list_dir_data=['20200219_223846','20200220_230830']
#list_dir_data=['20200222_233321','20200223_235457','20200224_234232']
#list_dir_data=['20200227_123416','20200226_200910']
#list_dir_data=['20200227_160031','20200226_200910']
#list_dir_data=['20200227_160031','20200228_130159']
#list_dir_data=['20200219_223846','20200220_230830','20200302_062250']
#list_dir_data=['20200218_212228','20200303_183303']
list_dir_data=['20200316_231639','20200316_231745','20200316_231815']