import React from 'react';
import Dropzone from 'react-dropzone-uploader';
import 'react-dropzone-uploader/dist/styles.css';

const Uploader = ({ project, ExplorerRef }) => {
  const getUploadParams = ({ file }) => {
    const body = new FormData();
    body.append('image', file);
    return { url: `/api/query_image?project_id=${project.id}`, body };
  };

  const predict = async (image_id) => {
    try {
      const response = await fetch(`/api/predict?image_id=${image_id}`, {
        method: 'GET'
      });
      if (!response.ok) {
        throw new Error('Prediction failed');
      }
      const data = await response.json();
      console.log('Predict response:', data);
    } catch (error) {
      console.error('Error predicting:', error);
    }
  };

  const handleChangeStatus = async ({ remove, xhr }, status) => {
    if (status === 'done') {
      console.log('file uploaded');
      remove();   // remove file once uploaded
      // xhr is response from POST /api/query_image
      const image_id = JSON.parse(xhr.response).image_id;
      ExplorerRef.addImage(image_id);
      // trigger prediction
      await predict(image_id);
      console.log('Image ID sent:', image_id);
    }
  };

  return (
    <Dropzone
      getUploadParams={getUploadParams}
      onChangeStatus={handleChangeStatus}
      // onSubmit
      accept="image/jpeg"
      maxFiles={100}
      multiple={true}
      maxSizeBytes={1000000} // 1MB
      autoUpload={true}
      inputContent='Drop jpg files here or click to browse (max 100 files, 1 MB size limit each file)'
      canCancel={true}
      styles={{
        dropzone: {
          minHeight: 200,
          maxHeight: 300
        },
        dropzoneActive: {
          borderColor: 'green'
        },
        inputLabel: {
          textAlign: 'center',
          fontSize: 16,
          fontWeight: 'normal'
        }
      }}
    />
  )
};

export default Uploader;
