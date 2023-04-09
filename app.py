from textwrap import wrap
import streamlit as st
import pandas as pd
import docx
import numpy as np
# from plc2audit import predict
from gptfunc import get_chatbot_response

def main():

    # choose input method of manual or upload file
    input_method = st.sidebar.radio('Input Method', ('Manual', 'Upload File'))
    # choose model
    model_name = st.sidebar.selectbox('选择模型', ['gpt-3.5-turbo', 'gpt-4'])

    if input_method == 'Manual':
        proc_text = st.text_area('Testing Procedures')

        if st.button('Predict'):
            # get prediction result
            result = get_chatbot_response(proc_text,model_name)
            # display result
            st.write(result)

    elif input_method == 'Upload File':
        # upload file
        upload_file = st.file_uploader('Upload workpaper',
                                       type=['xlsx', 'docx'])
        if upload_file is not None:
            # if upload file is xlsx
            if upload_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                # get sheet names list from excel file
                xls = pd.ExcelFile(upload_file)
                sheets = xls.sheet_names
                # choose sheet name and click button
                sheet_name = st.selectbox('Choose sheetname', sheets)

                # choose header row
                header_row = st.number_input('Choose header row',
                                             min_value=0,
                                             max_value=10,
                                             value=0)
                df = pd.read_excel(upload_file,
                                   header=header_row,
                                   sheet_name=sheet_name)
                # filllna
                df = df.fillna('')
                # display the first five rows
                st.write(df.astype(str))

                # get df columns
                cols = df.columns
                # choose proc_text and audit_text column
                proc_col = st.sidebar.selectbox('Choose procedure column',
                                                cols)
                # get proc_text and audit_text list
                proc_list = df[proc_col].tolist()

            elif upload_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                # read docx file
                document = docx.Document(upload_file)
                # get table data
                tablels = []
                for table in document.tables:
                    tb = []
                    for row in table.rows:
                        rl = []
                        for cell in row.cells:
                            rl.append(cell.text)
                        tb.append(rl)
                    tablels.append(tb)

                # get tablels index list
                tablels_index = list(range(len(tablels)))
                # choose tablels index
                tablels_no = st.selectbox('Choose table index', tablels_index)
                # choose header row
                header_row = st.number_input('Choose header row',
                                             min_value=0,
                                             max_value=10,
                                             value=0)

                if tablels_no is not None:
                    # get tablels
                    data = tablels[tablels_no]
                    dataarray = np.array(data)
                    dataarray2 = dataarray[header_row:, :]
                    df = pd.DataFrame(dataarray2)

                    st.write(df.astype(str))

                    # get df columns
                    cols = df.columns
                    # choose proc_text and audit_text column
                    proc_col = st.sidebar.selectbox('Choose procedure column',
                                                    cols)
                    # get proc_text and audit_text list
                    proc_list = df[proc_col].tolist()
                else:
                    st.error('No table found in the document')
                    proc_list = []
        else:
            st.error('No file selected')
            proc_list = []

        # top = st.sidebar.slider('AuditProc Generation Number',
        #                         min_value=1,
        #                         max_value=10,
        #                         value=3)
        # # choose max length of auditproc
        # max_length = st.sidebar.slider('Max Length of AuditProc',
        #                             min_value=25,
        #                             max_value=100,
        #                             value=50)
        # get proc_list and audit_list length
        proc_len = len(proc_list)

        # if proc_list or audit_list is empty or not equal
        if proc_len == 0:
            st.error('Procedure list must not empty')
            return
        else:
            # choose start and end index
            start_idx = st.sidebar.number_input('Choose start index',
                                                min_value=0,
                                                max_value=proc_len - 1,
                                                value=0)
            end_idx = st.sidebar.number_input('Choose end index',
                                            min_value=start_idx,
                                            max_value=proc_len - 1,
                                            value=proc_len - 1)

            # get proc_list and audit_list
            subproc_list = proc_list[start_idx:end_idx + 1]

        # display subproc_list
        st.subheader('Procedure List')
        # display subproc_list
        for i, proc in enumerate(subproc_list):
            st.markdown('##### ' + str(i) + ": " + proc)

        # input prompt text
        # prompt_text = st.sidebar.text_area('Prompt Text')

        search = st.sidebar.button('Convert')

        if search:
            # split list into batch of 5
            batch_num = 5
            proc_list_batch = [
                subproc_list[i:i + batch_num]
                for i in range(0, len(subproc_list), batch_num)
            ]

            dfls = []
            # get proc and audit batch
            for j, proc_batch in enumerate(proc_list_batch):

                with st.spinner('Converting...'):
                    # range of the batch
                    start = j * batch_num + 1
                    end = start + len(proc_batch) - 1

                    st.subheader('Results: ' + f'{start}-{end}')

                    # get audit list
                    # audit_batch = predict(proc_batch,8, top, max_length)
                    auditls = []
                    for i, proc in enumerate(proc_batch):
                        # audit range with stride x
                        # audit_start = i * top
                        # audit_end = audit_start + top
                        # get audit list
                        # audit_list = audit_batch[audit_start:audit_end]
                        # check if proc is blank, after remove space
                        if proc.replace(' ', '') == '':
                            st.error('Procedure must not empty')
                            audit_list = ''
                        else:
                            audit_list = get_chatbot_response(proc,model_name)
                        auditls.append(audit_list)
                        # get count number from start
                        count = str(j * batch_num + i + 1)
                        
                        # print proc and audit list
                        st.warning('Policy '+count+':')
                        st.write(proc)
                        # print audit list
                        st.info('Audit Procedure '+count+': ')
                        # for audit in audit_list:
                        #     st.write(audit)
                        st.write(audit_list)
                    # convert to dataframe
                    df = pd.DataFrame({
                        'policy': proc_batch,
                        'auditproc': auditls
                    })
                    dfls.append(df)

            # conversion is done
            st.sidebar.success('Conversion Finish')
            # if dfls not empty
            if dfls:
                alldf = pd.concat(dfls)
                # df explode by auditproc and reset index
                alldf = alldf.explode('auditproc')
                alldf = alldf.reset_index(drop=True)                    
                st.sidebar.download_button(label='Download',
                                        data=alldf.to_csv(),
                                        file_name='plc2auditresult.csv',
                                        mime='text/csv')


if __name__ == '__main__':
    main()