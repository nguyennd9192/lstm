from general_lib import *
import tensorflow as tf
from wg import WindowGenerator 
from model import *
from plot import *
from sklearn.preprocessing import MaxAbsScaler
import joblib
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Dense, LSTM, Lambda, Bidirectional, TimeDistributed
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from sklearn.preprocessing import MinMaxScaler
from keras.layers import ConvLSTM2D



def load_lwhp_data(train_data, test_data):
	

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

	features = ["high_temp", "demand", "low_temp"]
	for feature in features:
		fig = plt.figure(figsize=(9, 16), linewidth=0.8)
		years = range(2018, 2022) # list(set(df['year']))
		n_years = len(years)
		grid = plt.GridSpec(n_years, 4, hspace=0.3, wspace=1)

		for ith, year in enumerate(years):
			ax = fig.add_subplot(grid[ith, :]) # xticklabels=[], sharex=ax

			if ith == len(years) - 1:
				allyear_file = result_dir +"/LW_HP/{}_years/multiple.pdf".format(feature)

			else:
				allyear_file = None


			this_year_df = df[df['year'] == year]#.sort_values('timestamp_s')
			year_index = this_year_df.index
			print (this_year_df)
			
			saveat = result_dir +"/data/{}_years/{}.pdf".format(feature, year)
			coloraray, mappable = get_normcolor(c_value=range(len(year_index)), v_range=None)

			x = range(len(year_index))
			y = this_year_df[feature].values
			this_date = this_year_df['date'].values

			xtick_pos = range(1, 365, 30)
			xtick_name = [this_date[k] for k in xtick_pos]
			# scatter_plot(x=x, y=y, 
			# 	ax=None, xvline=None, yhline=None, 
			# 	sigma=None, mode='scatter-line', lbl=None, name=name, 
			# 	x_label='x', y_label=feature, 
			# 	save_file=saveat, interpolate=False, coloraray=coloraray, mappable=mappable,
			# 	xtick_pos=xtick_pos, xtick_name=xtick_name, 
			# 	linestyle='-.', marker='o', title=None)


			scatter_plot(x=x, y=y, 
				ax=ax, xvline=None, yhline=None, 
				sigma=None, mode='scatter-line', lbl=None, name=None, 
				x_label='x', y_label=feature, 
				save_file=allyear_file, interpolate=False, coloraray=coloraray, mappable=mappable,
				xtick_pos=xtick_pos, xtick_name=xtick_name, 
				linestyle='-.', marker='o', title=None)
	
	df.to_csv(input_dir + "LW_HP_process.csv")


	# # # # split train/test
	# # mask = df['demand'].isnull() # # for null ground truth
	n_test = 200
	mask = np.zeros(n_days, dtype=bool)
	mask[-n_test:] = True

	test_df = df[mask]
	train_df = df[~mask]


	pv = ["high_temp", "low_temp", "year", "month", "day",
		"Year_sin",	"Year_cos"]

	train_df = train_df[pv]
	test_df = test_df[pv]

	abs_scaler = MaxAbsScaler()
	abs_scaler.fit(train_df)
	scaled_train_data = abs_scaler.transform(train_df)
	scaled_test_data = abs_scaler.transform(test_df)

	scaled_train_df = pd.DataFrame(scaled_train_data, columns=train_df.columns)
	scaled_test_df = pd.DataFrame(scaled_test_data, columns=train_df.columns)

	scaled_train_df[label_columns] = df[~mask][label_columns]
	scaled_train_df.to_csv(train_data)
	scaled_test_df.to_csv(test_data)

	train_df[label_columns] = df[~mask][label_columns]
	test_df[label_columns] = df[mask][label_columns]

	train_df.to_csv(train_data)
	test_df.to_csv(test_data)

	return train_data, test_data



def call_model(model, savedir, train_df, val_df, test_df, out_steps):

	# # # for window size 24
	# for shift in range(1, 10, 2): # # [1,3,5,7,9] 

	# # function to deal with time-series data only
	# # independent with learning rate of lstm model
	this_sdir = savedir +"/lw_shift_{0}".format(shift)
	if os.path.exists(this_sdir):
		shutil.rmtree(this_sdir)

	wide_window = WindowGenerator(
		input_width=iw, label_width=out_steps, shift=out_steps,
		train_df=train_df, val_df=val_df, test_df=test_df,
		label_columns=label_columns)
	model.compile(loss=tf.keras.losses.MeanSquaredError(), # MeanSquaredError
			 metrics=[tf.keras.metrics.MeanAbsoluteError()])

	history = compile_and_fit(model, wide_window)
	print('Input shape: ', wide_window.example[0].shape)
	print('Output shape: ', model(wide_window.example[0]).shape)


	performance = model.evaluate(wide_window.test, verbose=0)
	print (performance)
	mae = round(performance[1], 3) 
	print ("Performance: ", performance)
	print ("======")
	
	fig =  plt.figure(figsize=(12, 6))

	wide_window.plot(model, plot_col=label_columns[0])
	plt.tight_layout(pad=1.1)
	plt.text(1, 0, "MAE: {}".format(mae), dict(size=30))

	saveat = this_sdir +"/wd{0}_lw{1}_{2}/prediction.pdf".format(iw, iw, mae)
	plt.title(saveat.replace(result_dir, ""))
	makedirs(saveat)
	plt.savefig(saveat)
	release_mem(fig)

	print ("======")

	y_pred = model.predict(wide_window.test, verbose=0)

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
	model_save = savedir + 'model.pkl'
	joblib.dump(model, model_save)
	return model_save

def train(train_data, savedir, model_type='base', out_steps=1):
	df = pd.read_csv(train_data, index_col=0)
	column_indices = {name: i for i, name in enumerate(df.columns)}

	# scaler = MinMaxScaler()
	# df[label_columns] = scaler.fit_transform(df[label_columns])
	df[label_columns] -= demand_baseline


	n = len(df)

	n_test = max(int(n*0.2), 450)
	
	n_train_val = len(df) - n_test
	n_val = max(int(n_train_val*0.3), 450)
	n_train = n_train_val - n_val
	assert n_train + n_val + n_test == n


	train_df = df[0:n_train]
	val_df = df[n_train:n_train+n_val]
	test_df = df[-n_test:]

	num_features = df.shape[1]

	# train_mean = train_df.mean()
	# train_std = train_df.std()

	# train_df = (train_df - train_mean) / train_std
	# val_df = (val_df - train_mean) / train_std
	# test_df = (test_df - train_mean) / train_std

	# df_std = (df - train_mean) / train_std
	# df_std = df_std.melt(var_name='Column', value_name='Normalized')
	# fig =  plt.figure(figsize=(12, 6))
	# ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
	# _ = ax.set_xticklabels(df.keys(), rotation=90)
	# plt.tight_layout(pad=1.1)

	# plt.savefig(result_dir +"/Normalized_feature.pdf")
	# release_mem(fig)


	# # # running baseline
	if model_type=='base':
		# # # for single step window
		single_step_window = WindowGenerator(
			input_width=1, label_width=1, shift=1,
			label_columns=label_columns,
			train_df=train_df, val_df=val_df, test_df=test_df,
			)

		for example_inputs, example_labels in single_step_window.train.take(1):
			print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
			print(f'Labels shape (batch, time, features): {example_labels.shape}')


		model = Baseline(label_index=column_indices[label_columns[0]])
		# model.compile(loss=tf.keras.losses.MeanSquaredError(),
  #               optimizer=tf.keras.optimizers.Adam(),
  #               metrics=[tf.keras.metrics.MeanAbsoluteError()])

		# val_performance = {}
		# performance = {}
		# val_performance['Baseline'] = model.evaluate(single_step_window.val)
		# performance['Baseline'] = model.evaluate(single_step_window.test, verbose=0)
	# # to show linear model prediction
	if model_type=='linear':
		model = tf.keras.Sequential([tf.keras.layers.Dense(units=1)])

	if model_type=='dense':
		model = tf.keras.Sequential([
		tf.keras.layers.Dense(units=64, activation='relu'),
		tf.keras.layers.Dense(units=64, activation='relu'),
		tf.keras.layers.Dense(units=1)
			])

	if model_type=='lstm':
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

	if model_type=='res-net':
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
	
	if model_type == 'multi-linear':
		model = tf.keras.Sequential()
		# 	# Shape [batch, time, features] => [batch, 1, features]
		model.add(Lambda(lambda x: x[:, -1:, :]))
		model.add(Dense(out_steps*num_features))

		# model.add(Dense(out_steps*num_features*2))
		# model.add(Dropout(rate=0.2))


		model.add(Dense(64, kernel_initializer=tf.initializers.zeros()))

		# model.add(Dropout(rate=0.3))
		# model.add(Dense(128, kernel_initializer=tf.initializers.zeros()))

		# model.add(Dense(1024))
		# model.add(Dropout(rate=0.3))

		model.add(Dense(out_steps*num_features))

		model.add(tf.keras.layers.Reshape([out_steps, num_features])) 		


	if model_type=='multi-lstm':
		params = {
					"dropout": 0.3,
					"lstm_units": 50,
				}
		model = tf.keras.Sequential()
		# 	# Shape [batch, time, features] => [batch, 1, features]
		model.add(Lambda(lambda x: x[:, -1:, :]))
		# model.add(Dense(out_steps*num_features, kernel_initializer=tf.initializers.zeros()))

		model.add( Bidirectional(LSTM(units=params["lstm_units"], return_sequences=True, 
					activation='relu',))		)
		# model.add(LSTM(units=params["lstm_units"], return_sequences=False))
		# model.add(Dropout(rate=params["dropout"]))

		model.add(Dense(out_steps*num_features))
		model.add(tf.keras.layers.Reshape([out_steps, num_features])) 		


	if model_type=='multi-cnn_lstm':
		model = tf.keras.Sequential()
		# model.add(Lambda(lambda x: x[:, -1:, :]))
		model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', 
			# input_shape=(n_seq, 1, n_steps, n_features)
			)
		)

		# model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu'), 
		# 	# input_shape=(None, 1, num_features)
		# 	)
		# )
		# model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
		# model.add(TimeDistributed(Flatten()))
		# model.add( Bidirectional(LSTM(units=50, activation='relu',)))
		model.add(Dense(out_steps*num_features))
		model.add(tf.keras.layers.Reshape([out_steps, num_features])) 		


	model_dir = call_model(model=model, savedir=savedir,
		train_df=train_df, val_df=val_df, test_df=test_df, out_steps=out_steps)





	return model_dir

def predict(model_dir, train_data, test_data, out_steps):
	df = pd.read_csv(train_data, index_col=0)
	test_df_org = pd.read_csv(test_data, index_col=0)
	# test_df_org[label_columns[0]] = None

	# scaler = MinMaxScaler()
	# df[label_columns] = scaler.fit_transform(df[label_columns])
	df[label_columns] -= demand_baseline


	n_train = int(len(df)*0.7)
	train_df = df[0:n_train]
	val_df = df[n_train:]

	tmp = [df[-iw:], test_df_org]
	test_df = pd.concat(tmp, ignore_index=False)

	model = joblib.load(model_dir)
	window = WindowGenerator(
		input_width=iw, label_width=out_steps, shift=out_steps,
		train_df=train_df, val_df=val_df, test_df=test_df,
		label_columns=label_columns)
	pred_data = window.test

	label_col_index = window.label_columns_indices.get(label_columns[0], None)
	
	predictions_scaled =  model.predict(pred_data)
	predictions = predictions_scaled[0, :, label_col_index] + demand_baseline
	test_df_org["predicted"] = predictions
	test_df_org.to_csv(test_data.replace(".csv", "_pred.csv"))


	print("elem pred", predictions)
	print ("shape:", predictions.shape)


	fig, ax = plt.subplots(figsize=(16, 9), linewidth=1.0) # 

	# plt.plot(window.input_indices, inputs[n, :, plot_col_index],
	# 		         label='Inputs', marker='.', zorder=-10)

	# # show inputs
	y_scaled = test_df[:iw][label_columns].values
	# y = scaler.inverse_transform(y_scaled)
	y = y_scaled + demand_baseline
	x = range(len(y))

	plt.scatter(x, y,  marker='.', color="blue", s=30, label='Most updated data')
	plt.plot(x, y, color="blue", linestyle="-")

	plt.scatter(window.label_indices, predictions,
			marker='X', edgecolors='k', label='Predictions',
			c='#ff7f0e', s=64)
	plt.plot(window.label_indices, predictions, 
			color="#ff7f0e", linestyle="-.")


	plt.scatter(window.label_indices, test_df_org[label_columns[0]], 
		color="black", marker='o', edgecolors='k', label='True value')
	plt.plot(window.label_indices, test_df_org[label_columns[0]],
		 color="black", linestyle="-")

	


	date =["{0}/{1}/{2}".format(test_df.loc[i, "year"], 
		test_df.loc[i, "month"], test_df.loc[i, "day"]) 
					for i in test_df.index ]
	xtick_pos = range(1, len(date), 30)
	xtick_name = [date[k] for k in xtick_pos]
	ax.set_xticks(xtick_pos)
	ax.set_xticklabels(xtick_name, rotation=40)
	ax.tick_params(axis='both', labelsize=16)
	plt.title("Prediction using {}".format(model_type))
	plt.legend()

	saveat = result_dir +"/predictions.pdf"
	makedirs(saveat)
	plt.savefig(saveat)
	release_mem(fig)



if __name__ == "__main__":
	train_data = input_dir + "LW_HP_train.csv"
	test_data = input_dir + "LW_HP_test.csv"
	model_type = 'multi-lstm' # 'multi-linear', 'multi-lstm', 'multi-cnn_lstm'
	result_dir += '/LW_HP/{}/'.format(model_type)

	# prepr_data = load_data()
	# train(prepr_data)


	# # for lwhp data
	# load_lwhp_data(train_data, test_data)

	# # base, linear, dens, res-net, lstm
	fc_df = pd.read_csv(test_data, index_col=0)
	out_steps = len(fc_df) 
	model_dir = train(train_data, savedir=result_dir, 
		model_type=model_type, out_steps=out_steps) 

	model_dir = result_dir + "/model.pkl"
	predict(model_dir=model_dir, train_data=train_data, test_data=test_data, 
		out_steps=out_steps)



































