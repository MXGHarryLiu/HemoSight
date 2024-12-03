import React from 'react';
import { Image, Stack } from 'react-bootstrap';
import { BRAND } from '../App.jsx';
import { Container } from 'react-bootstrap';

function WelcomePage() {

  return (
    <>
      <Container>
        <h2 className="text-center">Welcome</h2>

        <h3>Introduction</h3>
        <Stack>


          <Image src="https://placehold.co/600x50" fluid />
        </Stack>
        <h5>What?</h5>
        <p>
          {BRAND} is a web application that uses machine learning to analyze peripheral blood smear (PBS) images. Our aim is to enable hematopathologists to explore morphological landscape of white blood cells and provide foundation to AI-assisted clinical evaluation of PBS samples.   
        </p>
        <h5>Why?</h5>
        <p>
          Despite the availability of automated analyzers such as CellaVision, manual review remains central to clinical workflows to correct misclassified cell type labels, which can be subjective, qualitative, and time-consuming. {BRAND} uses self-supervised computer vision models to clasify WBC crops taken by automated analyzers such as Sysmex® DI-60, where the images are exported by CellaVision® software. We promote self-supervised learning that can make use of unlabeled data and adapt to different label sets. 
        </p>
        <h5>Who?</h5>
        <p>
          {BRAND} is developed by researchers at the University of Texas MD Anderson Cancer Center, lead by Dr. Yinyin Yuan under the division of pathology and laboratory medicine and institute of data science for oncology. Current developers include: Zhuohe Liu. 
          <br />
          To report bugs, request features, or ask questions, please contact developers through their MD Anderson emails or create an issue on <a href="https://github.com/MDA-Yuan-Lab/Hematology/issues" target="_blank" rel="noreferrer">GitHub</a>.
        </p>
        <h5>How?</h5>
        <p>
          Click 'Sign Up' to create an account or 'Login' to access the application. Once logged in, you can go to 'Input' page for training data distribution, 'Output' page for model performance comparison and training history, and 'Upload' page for uploading images and try out model inference.
        </p>
        <h5>Limitations</h5>
        <p>
          We are working to add the support for the following features:
          <ul>
            <li>Improve interface layout of the data explorer. </li>
            <li>Download prediction and summary report. </li>
            <li>Enable label correction in data explorer. </li>
          </ul>
        </p>
        <h5>Disclaimer</h5>
        <p>
          <span style={{ color:'red' }}>This application is not approved for clinical use or primary diagnosis.</span>
          <br />
          We are working with MD Anderson to ensure the safety and security of the data. All data are stored in secure MD Anderson servers.
          <br />
          We are a non-profit organization. Despite that we are actively maintaining the software, we cannot guarantee the availability of the service. In addition, we will try to acknowledge all the issue reported and feature requested, but we cannot guarantee the timeline of the development. For urgent issues, please contact the developers directly. We are not responsible for any loss of data or any other damages caused by the use of this software.
          <br />
          We are currently working to potentially open-source the application and accept external contributions. For now, please contact the developers if you want to contribute to the source code.
        </p>
        <h5>Acknowledgement</h5>
        <p>
          We thank Dr. Yinyin Yuan for her leadership and support. We thank Dr. Simon P. Castillo for many brainstorming meetings and discussion leading to this project. We thank hematopathologists Drs. Zhihong Hu, Xiaoping Sun, Xin Han in reviewing and annotating images for model training and providing feedback to the user interface. We gratefully acknowledge the funding and support provided by MD Anderson Cancer Center.
        </p>
        <h5>References</h5>
        <p>
          Datasets and annotations: 
          <ul>
            <li>Acevedo, A., Alférez, S., Merino, A., Puigví, L., & Rodellar, J. (2019). Recognition of peripheral blood cell images using convolutional neural networks. Computer Methods and Programs in Biomedicine, 180, 105020. <a href="https://doi.org/10.1016/J.CMPB.2019.105020" target="_blank" rel="noreferrer">doi:10.1016/J.CMPB.2019.105020</a></li>
            <li>Tsutsui, S., Pang, W., & Wen, B. (2023). WBCAtt: A White Blood Cell Dataset Annotated with Detailed Morphological Attributes. Advances in Neural Information Processing Systems, 36, 50796–50824. <a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/9f34484e5b8d87f09cc58c292a1c9f5d-Abstract-Datasets_and_Benchmarks.html" target="_blank" rel="noreferrer">Link</a></li>
          </ul>
          Methods:
          <ul>
            <li>Schroff, F., Kalenichenko, D., & Philbin, J. (2015). Facenet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 815-823).</li>
            <li>Poličar, P. G., Stražar, M., & Zupan, B. (2023). Embedding to reference t-SNE space addresses batch effects in single-cell classification. Machine Learning, 112(2), 721-740.</li>
          </ul>
        </p>
      </Container>
    </>
  );
}

export default WelcomePage;
