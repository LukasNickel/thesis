from aict_tools.io import (read_data, append_column_to_hdf5)
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PATH AND STUFF')
    parser.add_argument('data_path', type=str)
    parser.add_argument('column', type=str)
    args = parser.parse_args()

    a = read_data(args.data_path, 'array_events') 
    t = read_data(args.data_path, 'telescope_events') 
    new_t = t.merge(
        a[['run_id', 'array_event_id', args.column]],
        on=['run_id', 'array_event_id'],
        how='left')
    append_column_to_hdf5(args.data_path, new_t[args.column], 'telescope_events', args.column)
