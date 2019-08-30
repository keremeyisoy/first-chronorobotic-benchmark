# first-chronorobotic-benchmark
<br />
$pip install networkx<br />
$apt install libopencv-dev python-opencv<br />
$python run_testing_method.py<br />
<br /><br />

Before run this, please create new directories with your models name inside following directories; 'models' and 'results'
and, change the variable 'model' in run_testing_method.py with the same name of the directories you created.

You can find the list of positions in '/data/positions.txt' (x, y, angle)
Model outputs should be same as this file with an additional column of weights (order of rows is not important for testing method).

Here, are the parameters of grid;

edges of cells: x...0.5 [m], y...0.5 [m], angle...pi/4.0 [rad]<br />
number of cells: x...24, y...33, angles...8<br />
center of "first" cell: (-9.5, 0.25, -3.0*pi/4.0)<br />
center of "last" cell: (2.0, 16.25, pi) <br />

If you change the argument 'create_video' in run_testing_method.py to True, there will be video of every time window in results

outputs will be written in ../results/$model/output.txt in following format;<br />
list of values; [testing_time, number_of_detections_in_testing_data, interactions_of_dummy_model_clockwise, interactions_of_dummy_model_counterclockwise, interactions_of_real_model_clockwise, interactions_of_real_model_counterclockwise, total_weight_in_clockwise, total_weight_in_counterclockwise, total_interactions_of_chosen_trajectory]<br />
<br />
Since this code is prepared in a short time for scientific reasons, sorry in advance for any ambiguity
