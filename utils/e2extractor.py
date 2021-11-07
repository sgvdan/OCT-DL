from oct_converter.readers import E2E
import PySimpleGUI as sg
from PIL import Image, ImageEnhance
import shutil
import os
from pathlib import Path


def extract_e2e(src_e2e, dst_dir):
    src_e2e_path = Path(src_e2e)
    dst_dir_path = Path(dst_dir)

    # Sanity
    if src_e2e_path.suffix.lower() != '.e2e' or not src_e2e_path.is_file():
        return False

    # Copy original e2e volume & save each b-scan to 'images' dir
    images_path = dst_dir_path / 'images'
    os.makedirs(images_path, exist_ok=True)

    shutil.copy2(src_e2e_path, dst_dir_path)

    idx = 0
    for volume in E2E(src_e2e_path).read_oct_volume():
        for b_scan in volume.volume:
            img = Image.fromarray(b_scan).convert('RGB')
            enhancer = ImageEnhance.Brightness(img)
            img_enhanced = enhancer.enhance(2)
            img_enhanced.save(images_path / '{idx}_{pat_id}.tiff'.format(idx=idx, pat_id=volume.patient_id))
            idx += 1

    return True


def main():
    e2e_file_column = [
        [
            sg.Text('Enter .E2E file path:'),
            sg.In(size=(25,1), enable_events=True, key='-E2E_FILE-'),
            sg.FileBrowse(),
        ]
    ]

    dst_dir_column = [
        [
            sg.Text('Enter (empty) destination directory path:'),
            sg.In(size=(25,1), enable_events=True, key='-DST_DIR-'),
            sg.FolderBrowse(),
        ]
    ]
    layout = [
        [
            sg.Text("Welcome to the famous E2Extractor!")
        ],
        [
            sg.Column(e2e_file_column),
            sg.VSeperator(),
            sg.Column(dst_dir_column),
        ],
        [
            sg.Button("GO", key='-BUTTON-',  button_color=('white', 'blue')),
            sg.Button("EXIT", key='-EXIT-', button_color=('white', 'black')),

        ]
    ]

    # Create the window
    window = sg.Window("E2Extractor", layout)
    window.finalize()

    while True:
        event, values = window.read()

        # End program if user closes window
        if event == sg.WIN_CLOSED or event == '-EXIT-':
            break

        # OK. We can go!
        if event == "-BUTTON-":
            if extract_e2e(values['-E2E_FILE-'], values['-DST_DIR-']):
                window['-BUTTON-'].update('DONE',  button_color='green', disabled=True)
            else:
                window['-BUTTON-'].update('ERROR', button_color='red', disabled=True)

    window.close()

    print('done')


if __name__ == '__main__':
    main()