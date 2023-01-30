import pandas as pd
import os, glob, ntpath, pickle, functools, copy, sys, re, gc, math, shutil, random
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
main_dir = "/Users/nguyennguyenduong/Dropbox/Document/2022/MrToriyama/predict_price/" # "/home/nguyennd/work/lstm/"
input_dir = main_dir + "input/"
code_dir = main_dir + "code/"
result_dir = main_dir + "result/"

label_columns = ['demand']

shift = 1
iw = 100
demand_baseline = 200

def makedirs(file):
	if not os.path.isdir(os.path.dirname(file)):
		os.makedirs(os.path.dirname(file))


def release_mem(fig):
    fig.clf()
    plt.close()
    gc.collect()



def load_data():
	zip_path = tf.keras.utils.get_file(
		origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
		fname='jena_climate_2009_2016.csv.zip', extract=True)
	csv_path, _ = os.path.splitext(zip_path)

	df = pd.read_csv(csv_path)

	# Slice [start:stop:step], starting from index 5 take every 6th record.
	df = df[5::6]
	date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
	df.head()

	plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
	plot_features = df[plot_cols]
	plot_features.index = date_time
	_ = plot_features.plot(subplots=True)

	plot_features = df[plot_cols][:480]
	plot_features.index = date_time[:480]

	fig = plt.figure(figsize=(8, 8))
	_ = plot_features.plot(subplots=True)

	makedirs(result_dir)
	plt.savefig(result_dir +"/jena_climate_2009_2016.pdf")
	release_mem(fig)


	# # remove bad data
	wv = df['wv (m/s)']
	bad_wv = wv == -9999.0
	wv[bad_wv] = 0.0

	max_wv = df['max. wv (m/s)']
	bad_max_wv = max_wv == -9999.0
	max_wv[bad_max_wv] = 0.0

	# The above inplace edits are reflected in the DataFrame.
	# df['wv (m/s)'].min()

	# plt.hist2d(df['wd (deg)'], df['wv (m/s)'], bins=(50, 50), vmax=400)
	# plt.colorbar()
	# plt.xlabel('Wind Direction [deg]')
	# plt.ylabel('Wind Velocity [m/s]')

	wv = df.pop('wv (m/s)')
	max_wv = df.pop('max. wv (m/s)')

	# Convert to radians.
	wd_rad = df.pop('wd (deg)')*np.pi / 180

	# Calculate the wind x and y components.
	df['Wx'] = wv*np.cos(wd_rad)
	df['Wy'] = wv*np.sin(wd_rad)

	# Calculate the max wind x and y components.
	df['max Wx'] = max_wv*np.cos(wd_rad)
	df['max Wy'] = max_wv*np.sin(wd_rad)

	fig, ax = plt.subplots(figsize=(9, 8), linewidth=1.0) # 

	plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
	plt.colorbar()
	plt.xlabel('Wind X [m/s]')
	plt.ylabel('Wind Y [m/s]')
	ax = plt.gca()
	ax.axis('tight')

	plt.savefig(result_dir +"/WindX_WindY.pdf")
	release_mem(fig)

	# # transform time
	timestamp_s = date_time.map(pd.Timestamp.timestamp)
	day = 24*60*60
	year = (365.2425)*day

	df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
	df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
	df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
	df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

	fig = plt.figure(figsize=(8, 8))
	plt.plot(np.array(df['Day sin'])[:25])
	plt.plot(np.array(df['Day cos'])[:25])
	plt.xlabel('Time [h]')
	plt.title('Time of day signal')
	plt.savefig(result_dir +"/time_of_day.pdf")
	release_mem(fig)

	prepr_data = result_dir +"/jena_climate_2009_2016.csv"
	df.to_csv(prepr_data)
	return prepr_data