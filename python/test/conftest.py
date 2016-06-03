

def fill_repository(repo, begin_end_cam_id):
    for begin, end, cam_id in begin_end_cam_id:
        params = begin, end, cam_id, 'bbb'
        repo._create_file_and_symlinks(*params)
