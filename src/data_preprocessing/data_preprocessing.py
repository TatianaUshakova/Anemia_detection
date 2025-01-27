import pandas as pd

def preprocess_original_data(
    img_table, hem_table, 
    column_image_names, sheet_num_imgs, id_imgs_column = 'Blood_Sample_ID', drop_columns_images=['Unnamed: 0', 'Hb_Value', 'Gender', 'Age_in_years', 'Image_Path'],
    column_hemoglobin='Haemoglobin (in mg/dl)', id_hem_column = 'Blood Sample ID', drop_columns_hem=['Unnamed: 0', 'index', 'Total Serial Number', 'S No.', 'Unique ID']
    ):
    """
    Preprocesses the original dataset by merging image data with hemoglobin data.

    This function merges two tables based on a common patient ID column, removes unnecessary 
    columns, and handles missing or invalid data. The default names are from original dataset naming, 
    these columns are dropped since they dont have value for predicting modeling.

    Args:
    - img_table (str): Path to the Excel file containing image paths and patient IDs.
    - hem_table (str): Path to the Excel file containing hemoglobin data and patient IDs.
    - column_image_names (str): Column name in the image table that contains image file names.
    - sheet_num_imgs (int): Sheet number in the image table containing data for the body part of interest. (for excel on multiple pages)
    - id_imgs_column (str, optional): Column name for patient IDs in the image table. Default is "Blood_Sample_ID".
    - drop_columns_images (list, optional): List of columns to drop from the image table. 
    - column_hemoglobin (str, optional): Column name for hemoglobin values in the hemoglobin table. 
    - id_hem_column (str, optional): Column name for patient IDs in the hemoglobin table. Default is "Blood Sample ID".
    - drop_columns_hem (list, optional): List of columns to drop from the hemoglobin table. 

    Returns:
    - pd.DataFrame: A merged DataFrame containing:
        - "Images": File names of patient images.
        - "Hemoglobin": Corresponding hemoglobin levels.
        - Other columns from the input tables, excluding the dropped columns.

    Notes:
    - Rows with missing "Images" values are removed.
    - This function assumes that patient IDs are consistent across both tables and merges across this column.
    """
    
    #reading info about images locations
    img_df = pd.read_excel(img_table, sheet_name=sheet_num_imgs)
    img_df = img_df.drop(columns=drop_columns_images, errors='ignore')
    img_df.rename(columns = {id_imgs_column:id_hem_column}, inplace=True)#rename consistently id columns to merge across it

    #reading info about patients hemoglobin
    hem_info = pd.read_excel(hem_table, sheet_name=0)
    hem_info = hem_info.drop(columns=drop_columns_hem, errors='ignore')

    #merge two previous datasets to one dataset
    patients_df = pd.merge(hem_info, img_df, on=id_hem_column, how='outer')
    patients_df.rename(columns = {column_hemoglobin:'Hemoglobin'}, inplace=True)
    patients_df.rename(columns = {column_image_names:'Images'}, inplace=True)

    # Drop rows where 'Hemoglobin' is NaN
    patients_df = patients_df.dropna(subset=['Images'])
    patients_df.reset_index(drop=True, inplace=True)

    return patients_df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('img_table', type = str, help = 'Path to excel table with image pathes of body parts of interest')
    parser.add_argument('hem_table', type = str, help = 'Path to excel table with patients hemoglobin data')
    parser.add_argument('column_image_names', type = str, help = 'Name of the column in img_table containing the image paths')
    parser.add_argument('sheet_num_imgs', type = int, help = 'Sheet number in img_table containing the images of body parts of imterest')

    args = parser.parse_args()
    patients_df = preprocess_original_data(
        args.img_table, args.hem_table, args.column_image_names, args.sheet_num_imgs)

    patients_df.to_excel('Merged_df.xlsx', index=False)