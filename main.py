from general_lib import *
import tensorflow as tf
from wg import WindowGenerator 
from model import *
from plot import *




def load_lwhp_data():
	

	filename = input_dir + "LW_HP.csv"
	df = pd.read_csv(filename)
	df['year'] = pd.DatetimeIndex(df['date']).year
	df['month'] = pd.DatetimeIndex(df['date']).month
	df['day'] = pd.DatetimeIndex(df['date']).day


	indexes = (df.index)
	n_days = len(df['year'])
	date_time = pd.to_datetime(df['date'], dayfirst=False)
	print (df.head())
	timestamp_s = date_time.map(pd.Timestamp.timestamp)

	df['timestamp_s'] = timestamp_s
	day = 24*60*60
	year = 365.2425 * day

	df['Year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
	df['Year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))

	# # # split train/test
	mask = df['demand'].isnull()
	test_df = df[mask]
	train_df = df[~mask]

	train_data = input_dir + "LW_HP_train.csv"
	test_data = input_dir + "LW_HP_test.csv"

	train_df.to_csv(train_data)
	test_df.to_csv(test_data)


	id_test = np.concatenate([	[k, 365*3+k]
			for k in range(1, 365, 30)
		 ])

	# # test convert
	fig = plt.figure(figsize=(8, 8))
	plt.plot(np.array(df['Year_sin']))
	plt.plot(np.array(df['Year_cos']))

	name = [None] * n_days

	for it in id_test:
		this_sin = df.loc[it, 'Year_sin']
		this_date = df.loc[it, 'date']
		plt.text(it, this_sin, this_date)
		name[it] = this_date

	plt.ylabel('Time (convert)')
	plt.title('Time of day signal')
	saveat = result_dir +"/LW_HP/time_of_day.pdf"
	makedirs(saveat)
	plt.savefig(saveat)
	release_mem(fig)

	# # # specify xtickname
	# xtick_pos = np.arange(-1.0, 1.1, 0.25)
	# xtick_name = []
	# oneyear_df = df[:365]
	# oneyear_idx = oneyear_df.index
	# ys_round = np.round(oneyear_df['Year_sin'], 2)
	# for s in xtick_pos:
	# 	tmp = oneyear_idx[ys_round==s][0]
	# 	this_date = str(oneyear_df.loc[tmp, 'date'])
	# 	md = this_date[:this_date.rfind("/")]
	# 	xtick_name.append(md)
	# coloraray, mappable = get_normcolor(c_value=this_year_df['date'], v_range=None)

	# # ind
	# # # merge
	fig = plt.figure(figsize=(9, 16), linewidth=0.8)
	years = range(2018, 2022) # list(set(df['year']))
	n_years = len(years)
	grid = plt.GridSpec(n_years, 4, hspace=0.3, wspace=1)


	for ith, year in enumerate(years):
		if ith==0:
			ax = fig.add_subplot(grid[ith, :]) # xticklabels=[], sharex=ax
			allyear_file = None

		elif ith == len(years) - 1:
			ax = fig.add_subplot(grid[ith, :])
			allyear_file = result_dir +"/LW_HP/demand_years/multiple.pdf"

		else:
			ax = fig.add_subplot(grid[ith, :])
			allyear_file = None


		this_year_df = train_df[train_df['year'] == year].sort_values('timestamp_s')
		year_index = this_year_df.index
		print (this_year_df)
		
		saveat = result_dir +"/LW_HP/demand_years/{}.pdf".format(year)
		coloraray, mappable = get_normcolor(c_value=range(len(year_index)), v_range=None)

		x = range(len(year_index))
		y = this_year_df['demand'].values
		this_date = this_year_df['date'].values

		xtick_pos = range(1, 365, 30)
		xtick_name = [this_date[k] for k in xtick_pos]
		# scatter_plot(x=x, y=y, 
		# 	ax=None, xvline=None, yhline=None, 
		# 	sigma=None, mode='scatter-line', lbl=None, name=name, 
		# 	x_label='x', y_label='demand', 
		# 	save_file=saveat, interpolate=False, coloraray=coloraray, mappable=mappable,
		# 	xtick_pos=xtick_pos, xtick_name=xtick_name, 
		# 	linestyle='-.', marker='o', title=None)


		scatter_plot(x=x, y=y, 
			ax=ax, xvline=None, yhline=None, 
			sigma=None, mode='scatter-line', lbl=None, name=None, 
			x_label='x', y_label='demand', 
			save_file=allyear_file, interpolate=False, coloraray=coloraray, mappable=mappable,
			xtick_pos=xtick_pos, xtick_name=xtick_name, 
			linestyle='-.', marker='o', title=None)
	
	df.to_csv(input_dir + "LW_HP_process.csv")
	return train_data, test_data


def call_model(model, savedir, train_df, val_df, test_df):

	# # # for window size 24
	# for shift in range(1, 10, 2): # # [1,3,5,7,9] 

	# # function to deal with time-series data only
	# # independent with learning rate of lstm model
	shift = 1
	this_sdir = savedir +"/lw_shift_{0}".format(shift)
	if os.path.exists(this_sdir):
		shutil.rmtree(this_sdir)

	# for iw in range(12, 25, 1): # 13 ==> 13*5 = 65
	iw = 12
	wide_window = WindowGenerator(
		input_width=iw, label_width=iw, shift=shift,
		train_df=train_df, val_df=val_df, test_df=test_df,
		label_columns=['T (degC)'])
	model.compile(loss=tf.keras.losses.MeanSquaredError(),
			 metrics=[tf.keras.metrics.MeanAbsoluteError()])

	history = compile_and_fit(model, wide_window)

	# print ("Test wide_window index:", wide_window.example[0])

	print('Input shape: ', wide_window.example[0].shape)
	print('Output shape: ', model(wide_window.example[0]).shape)




	performance = model.evaluate(wide_window.test, verbose=0)
	mae = round(performance[1], 3) 
	print ("Performance: ", performance)
	print ("======")
	
	fig =  plt.figure(figsize=(12, 6))

	wide_window.plot(model)
	plt.tight_layout(pad=1.1)
	plt.text(1, 0, "MAE: {}".format(mae), dict(size=30))

	saveat = this_sdir +"/wd{0}_lw{1}_{2}/prediction.pdf".format(iw, iw, mae)
	plt.title(saveat.replace(result_dir, ""))
	makedirs(saveat)
	plt.savefig(saveat)
	release_mem(fig)

	# fig =  plt.figure(figsize=(12, 6))
	# plt.bar(x = range(len(train_df.columns)),
	#         height=model.layers[0].kernel[:,0].numpy())
	# axis = plt.gca()
	# axis.set_xticks(range(len(train_df.columns)))
	# _ = axis.set_xticklabels(train_df.columns, rotation=90)
	# plt.tight_layout(pad=1.1)
	# saveat = this_sdir +"/wd{0}_lw{1}_{2}/linear_weight.pdf".format(iw, iw, mae)
	# plt.title(saveat.replace(result_dir, ""))
	# plt.savefig(saveat)
	# release_mem(fig)


def train(train_data):
	df = pd.read_csv(train_data, index_col=0)
	column_indices = {name: i for i, name in enumerate(df.columns)}

	n = len(df)
	train_df = df[0:int(n*0.7)]
	val_df = df[int(n*0.7):int(n*0.9)]
	test_df = df[int(n*0.9):]

	num_features = df.shape[1]


	train_mean = train_df.mean()
	train_std = train_df.std()

	train_df = (train_df - train_mean) / train_std
	val_df = (val_df - train_mean) / train_std
	test_df = (test_df - train_mean) / train_std

	df_std = (df - train_mean) / train_std
	df_std = df_std.melt(var_name='Column', value_name='Normalized')
	fig =  plt.figure(figsize=(12, 6))
	ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
	_ = ax.set_xticklabels(df.keys(), rotation=90)
	plt.tight_layout(pad=1.1)

	plt.savefig(result_dir +"/Normalized_feature.pdf")
	release_mem(fig)



	

	## # mismatch between indentation definition

	is_base = False
	is_ln = False 
	is_dense = True
	is_lstm = False
	is_res_net = False

	# # # running baseline
	if is_base:
		savedir = result_dir+"/bs/" 

		# # # for single step window
		single_step_window = WindowGenerator(
			input_width=1, label_width=1, shift=1,
			label_columns=['T (degC)'],
			train_df=train_df, val_df=val_df, test_df=test_df,
			)

		print (single_step_window)
		for example_inputs, example_labels in single_step_window.train.take(1):
			print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
			print(f'Labels shape (batch, time, features): {example_labels.shape}')


		model = Baseline(label_index=column_indices['T (degC)'])
		model.compile(loss=tf.keras.losses.MeanSquaredError(),
						 metrics=[tf.keras.metrics.MeanAbsoluteError()])

		val_performance = {}
		performance = {}
		val_performance['Baseline'] = model.evaluate(single_step_window.val)
		performance['Baseline'] = model.evaluate(single_step_window.test, verbose=0)
	# # to show linear model prediction
	if is_ln:
		savedir = result_dir+"/ln/"
		model = tf.keras.Sequential([tf.keras.layers.Dense(units=1)])

	if is_dense:
		savedir = result_dir+"/dense/"
		model = tf.keras.Sequential([
		tf.keras.layers.Dense(units=64, activation='relu'),
		tf.keras.layers.Dense(units=64, activation='relu'),
		tf.keras.layers.Dense(units=1)
			])

	if is_lstm:
		savedir = result_dir+"/lstm/"
		model = tf.keras.models.Sequential([
				# Shape [batch, time, features] => [batch, time, lstm_units]
				tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2,
							recurrent_dropout=0.0),

				# Shape => [batch, time, features]
				tf.keras.layers.Dense(units=1)
			]) 
			# # best MAE: 0.057, dropout=0.0
			# # best MAE: 0.05 , dropout=0.1

		# real_data_df = get_real_datastream()
		# pred_temp = model.predict(real_data_df[df.columns])

	if is_res_net:
		savedir = result_dir+"/resn/"

		model = ResidualWrapper(
			tf.keras.Sequential([
			tf.keras.layers.LSTM(32, return_sequences=True),
			tf.keras.layers.Dense(
				num_features,
				# The predicted deltas should start small.
				# Therefore, initialize the output layer with zeros.
				kernel_initializer=tf.initializers.zeros())
				]))
	call_model(model=model, savedir=savedir,
		train_df=train_df, val_df=val_df, test_df=test_df)



if __name__ == "__main__":

	# prepr_data = load_data()
	# train(prepr_data)

	train_data, test_data = load_lwhp_data()
	train(train_data)

	# # template data: prepr_data = result_dir +"/jena_climate_2009_2016.csv"
