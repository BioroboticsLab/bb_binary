"""Setup for tests and fixtures."""
import warnings
import pandas as pd

# enable stack trace for annoying SettingWithCopyWarning
warnings.simplefilter("error", pd.core.common.SettingWithCopyWarning)


def fill_repository(repo, begin_end_cam_id):
    for begin, end, cam_id in begin_end_cam_id:
        params = begin, end, cam_id, 'bbb'
        repo._create_file_and_symlinks(*params)
        fname = repo._get_filename(*params)
        with open(fname, 'w') as f:
            f.write(str(begin))
