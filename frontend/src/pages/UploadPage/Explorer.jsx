import React from 'react';
import { ButtonToolbar, ButtonGroup, Button, ToggleButton, Stack, Container, Row, Col, Tabs, Tab, Image, Form } from 'react-bootstrap';
import Thumbnail from './Thumbnail.jsx';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faArrowsRotate, faTrash, faUpload } from '@fortawesome/free-solid-svg-icons'
import DeleteImageModal from './DeleteImageModal.jsx';
import embed from 'vega-embed';
import * as vega from 'vega';
import { fetchVegaPlot } from '../../components/FigureCard.jsx';
import Uploader from './Uploader.jsx';

class Explorer extends React.Component {
  // props: project
  constructor(props) {
    super(props);
    this.state = {
      // fields of image:
      // image_id (string), filename (string), selected (bool), 
      // current (bool), label (list of strings), prob (list of floats),
      // status (string), reference_label (string)
      images: [],
      selection_enabled: false,
      vegaTemplate: { label: null, knn: null, summary: null, queryProbability: null},
      vegaView: { summary: null },
      tab: "details",
      showUploader: false,
      showFooter: false,
      showProbability: false
    };
    this.DeleteImageModalRef = React.createRef();
  }

  updateImage = async (image_id) => {
    const imgData = await this.#getImageData(image_id);
    this.setState(prevState => {
      const { images } = prevState;
      const imageIndex = images.findIndex(image => image.image_id === image_id);
      if (imageIndex === -1) {
        return prevState;
      } else {
        const updatedImages = [...images];
        updatedImages[imageIndex] = {
          ...updatedImages[imageIndex],
          label: imgData.label,
          prob: imgData.prob,
          status: imgData.status,
          reference_label: imgData.reference_label
        };
        return { images: updatedImages };
      }
    });
  };

  addImage = async (image_id) => {
    const imgData = await this.#getImageData(image_id);
    this.setState(prevState => {
      const { images } = prevState;
      const updatedImages = [...images];
      updatedImages.push({
        image_id: image_id,
        filename: imgData.filename,
        selected: false,
        current: false,
        label: imgData.label,
        prob: imgData.prob,
        status: imgData.status,
        reference_label: imgData.reference_label
      });
      return { images: updatedImages };
    });
  };

  #getImageData = async (image_id) => {
    const accessToken = sessionStorage.getItem('access_token');
    const response = await fetch(`/api/query_result/${image_id}`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${accessToken}`
      }
    });
    const data = await response.json();
    let label = [];
    let prob = [];
    let reference_label = "";
    if (data.status === "done") {
      label = data.label;
      prob = data.probability;
      reference_label = data.reference_label;
    }
    return {
      filename: data.filename,
      label: label,
      prob: prob,
      status: data.status,
      reference_label: reference_label
    };
  };

  toggleSelected = (image_id) => {
    if (this.state.selection_enabled) {
      // change selection
      this.setState(prevState => ({
        images: prevState.images.map(image => {
          if (image.image_id === image_id) {
            if (image.selected) {
              console.log('deselect', image_id);
            } else {
              console.log('select', image_id);
            }
            return { ...image, selected: !image.selected };
          }
          return image;
        })
      }));
    } else {
      // change current
      this.setState(prevState => ({
        images: prevState.images.map(image => {
          if (image.image_id === image_id) {
            return { ...image, current: true };
          } else {
            return { ...image, current: false };
          }
        }
        )
      }));
    }
  };

  #deleteSelectedImage = () => {
    // single image
    const selectedImage = this.state.images.find(image => image.selected);
    const image_id = selectedImage.image_id;
    this.DeleteImageModalRef.current.show(image_id);
  };

  deleteSelectedImageFinished = (image_id) => {
    this.setState(prevState => {
      const { images } = prevState;
      const updatedImages = images.filter(image => image.image_id !== image_id);
      return { images: updatedImages };
    });
  }

  toggleSelectionEnabled = (event) => {
    this.setState({ selection_enabled: event.currentTarget.checked });
  }

  #tabSelect = (k, image_id) => {
    const image = this.state.images.find(image => image.image_id === image_id);
    if (k === "details") {
      this.#renderProbabiltyPlot(image_id, image.label, image.prob);
    } else if (k === "knn") {
      this.#renderKNNPlot(image_id);
    } else if (k === "summary") {
      if (this.state.showProbability) {
        this.#renderQueryProbabilityPlot();
      } else {
        this.#renderSummaryPlot();
      }
    }
  }

  #renderProbabiltyPlot = async (image_id, label, prob) => {
    const values = label.map((label, i) => {
      return { label: label, probability: prob[i] };
    });
    let plot = this.state.vegaTemplate.label;
    plot.data.values = values; // update data
    //plot.usermeta.embedOptions.downloadFileName = `label probability ${image_id}`;
    const result = await embed('#probability-plot', plot);
    // result.vgSpec.usermeta.embedOptions.downloadFileName = `label probability ${image_id}`;
    // await result.view.resize().runAsync();
    // const tempdata = [{
    //   "label": "A", "probability": true
    // }];
    // const changeset = vega.changeset().remove(() => true).insert(tempdata);
    // await result.view.change('data', changeset).runAsync();
  }

  #renderKNNPlot = (image_id) => {
    let plot = this.state.vegaTemplate.knn;
    plot.data.url = `/api/query_knn?image_id=${image_id}`;
    plot.usermeta.embedOptions.downloadFileName = `knn query ${image_id}`;
    embed('#knn-plot', plot);
  }

  #renderSummaryPlot = async () => {
    let view = this.state.vegaView.summary
    if (view === null) {
      const plot = this.state.vegaTemplate.summary;
      view = await embed('#summary-plot', plot);
      this.setState({ vegaView: { summary: view } });
    }
    const { images } = this.state;
    const data = images.map(image => {
      return { label: image.label[0], selected: image.selected };
    });
    const changeset = vega.changeset().remove(() => true).insert(data);
    await view.view.change('data', changeset).runAsync();
  }

  #renderQueryProbabilityPlot = async () => {
    let view = this.state.vegaView.summary;
    if (view === null) {
      const plot = this.state.vegaTemplate.queryProbability;
      view = await embed('#query-probability-plot', plot);
      this.setState({ vegaView: { summary: view } });
    }
    const { images } = this.state;
    // create [{"category":"A", "probability": 0.1},{"category":"B", "probability": 0.1}...]
    const data = images.map(image => {
      // lopp thru image.label and image.prob
      const label = image.label;
      const prob = image.prob;
      const result = [];
      for (let i = 0; i < label.length; i++) {
        result.push({ category: label[i], probability: prob[i] });
      }
      return result;
    });
    // flatten the array
    const plotData = data.flat();
    const changeset = vega.changeset().remove(() => true).insert(plotData);
    await view.view.change('data', changeset).runAsync();
  }

  #summaryPlotSwitched = () => {
    this.setState({ showProbability: !this.state.showProbability, 
                    vegaView: { summary: null }});
  }

  componentDidMount() {
    this.#fetchProjectImages();
    if (this.state.vegaTemplate.label === null) {
      this.#fetchVegaTemplate();
    }
  }

  componentDidUpdate(prevProps, prevState) {
    const { images } = this.state;
    const currentImage = images.find(image => image.current);
    const prevImage = prevState.images.find(image => image.current);
    // whether the current image has changed
    if (prevImage && currentImage === undefined) {
      // if current image is cleared
      this.setState({ showFooter: false, vegaView: { summary: null } });
    }
    if (currentImage && (prevImage === undefined || currentImage.image_id !== prevImage.image_id)) {
      // if current image is done
      const image_id = currentImage.image_id;
      this.#tabSelect(this.state.tab, image_id);
      if (currentImage.status === "done") {
        this.setState({ showFooter: true });
      } else {
        this.setState({ showFooter: false, vegaView: { summary: null } });
      }
    }
    // whether tab has changed
    if (prevState.tab !== this.state.tab) {
      this.#tabSelect(this.state.tab, currentImage.image_id);
    }
    // whether selection has changed
    const selection_count = images.filter(image => image.selected).length;
    const prev_selection_count = prevState.images.filter(image => image.selected).length;
    if (selection_count !== prev_selection_count && this.state.tab === "summary") {
      // if not showProbability
      if (!this.state.showProbability) {
        this.#renderSummaryPlot();
      }
    }
    // whether probability plot has changed
    if (this.state.showProbability !== prevState.showProbability && this.state.tab === "summary") {
      if (this.state.showProbability) {
        this.#renderQueryProbabilityPlot();
      } else {
        this.#renderSummaryPlot();
      }
    }
    console.log('render')
  }

  #fetchProjectImages = async () => {
    const { project } = this.props;
    const response = await fetch(`/api/query_image_project/${project.id}`)
    const data = await response.json();
    const updatedImages = [];
    for (const image_id of data) {
      const imgData = await this.#getImageData(image_id);
      updatedImages.push({
        image_id: image_id,
        filename: imgData.filename,
        selected: false,
        current: false,
        label: imgData.label,
        prob: imgData.prob,
        status: imgData.status,
        reference_label: imgData.reference_label
      });
    }
    this.setState({ images: updatedImages, vegaView: { summary: null } });
  };

  #fetchVegaTemplate = async () => {
    this.setState({
      vegaTemplate: {
        label: await fetchVegaPlot('label_probability.lite'),
        knn: await fetchVegaPlot('query_knn.lite'),
        summary: await fetchVegaPlot('query_counts.lite'),
        queryProbability: await fetchVegaPlot('query_probability.lite')
      }
    });
  }

  render() {
    const { images, selection_enabled, showUploader, showFooter } = this.state;
    const hasSelected = images.filter(image => image.selected).length > 0;
    const hasOneSelected = images.filter(image => image.selected).length === 1;
    const currentImage = images.find(image => image.current);

    return (
      <>
        <Stack gap={1}>
          <p>Total {images.length} images. </p>
          <ButtonToolbar className="mb-2">
            <ButtonGroup className="mr-2">
              <ToggleButton id="show-uploader"
                type="checkbox"
                title={showUploader ? "Hide uploader" : "Show uploader"}
                checked={showUploader}
                variant="outline-primary"
                value="1"
                onChange={() => this.setState({ showUploader: !showUploader })}
              >
                <FontAwesomeIcon icon={faUpload} />
              </ToggleButton>
              <Button onClick={this.#fetchProjectImages}
                title="Refresh">
                <FontAwesomeIcon icon={faArrowsRotate} />
              </Button>
              <Button onClick={this.#deleteSelectedImage}
                disabled={!hasOneSelected}
                variant='danger'
                title="Delete selected image">
                <FontAwesomeIcon icon={faTrash} />
              </Button>
              <ToggleButton id="selection_enabled"
                type="checkbox"
                title="Select images to delete"
                checked={selection_enabled}
                variant="outline-primary"
                value="1"
                onChange={this.toggleSelectionEnabled}>
                Select
              </ToggleButton>
            </ButtonGroup>
          </ButtonToolbar>
          {showUploader &&
            <Uploader project={this.props.project}
              ExplorerRef={this} />
          }
          <div className="thumbnail-container border">
            {images.length > 0 ? (
              images.map(image => (
                <Thumbnail parent={this}
                  key={image.image_id}
                  image={image} />
              ))
            ) : (
              <p>No images</p>
            )
            }
          </div>
          {showFooter && currentImage !== undefined ? (
            <Tabs
              id="image-details"
              activeKey={this.state.tab}
              onSelect={(k) => this.setState({ tab: k })}
              className="mb-3">
              <Tab eventKey="details" title="Details">
                <Container>
                  <Row>
                    <Col md={6}>
                      <Stack gap={2}>
                        <div>ID: {currentImage.image_id}</div>
                        <div>Filename: {currentImage.filename}</div>
                        <div>Label inferred from filename: <b>{currentImage.reference_label}</b>.</div>
                        <div>Predicted label: <b>{currentImage.label[0]}</b>.</div>
                        <Image rounded src={`/api/query_image/${currentImage.image_id}`}
                          title={`${currentImage.filename} (${currentImage.image_id})`}
                          alt={`${currentImage.filename} (${currentImage.image_id})`} />
                      </Stack>
                    </Col>
                    <Col md={6}>
                      <Stack gap={3}>
                        <div id="probability-plot"></div>
                        <div>{this.state.vegaTemplate.label.usermeta.help}</div>
                      </Stack>
                    </Col>
                  </Row>
                </Container>
              </Tab>
              <Tab eventKey="knn" title="KNN">
                <Stack gap={3}>
                  <div id="knn-plot"></div>
                  <div>{this.state.vegaTemplate.knn.usermeta.help}</div>
                </Stack>
              </Tab>
              <Tab eventKey="summary" title="Summary">
                <Stack gap={3}>
                  <Form>
                    <Form.Check type="switch" id="switch-summary"
                      label="Show probability"
                      checked={this.state.showProbability}
                      onChange={this.#summaryPlotSwitched}
                    />
                  </Form>
                  {this.state.showProbability ? (
                    <>
                      <div id="query-probability-plot"></div>
                      <div>{this.state.vegaTemplate.queryProbability.usermeta.help}</div>
                    </>
                  ) : (
                    <>
                      <div id="summary-plot"></div>
                      <div>{this.state.vegaTemplate.summary.usermeta.help}</div>
                    </>
                  )}
                </Stack>
              </Tab>
            </Tabs>
          ) : (
            <div className="border text-muted p-3">
              Click on thumbnails to load details.
            </div>
          )}
        </Stack>
        <DeleteImageModal
          parent={this}
          ref={this.DeleteImageModalRef}
        />
      </>
    );
  }
}

export default Explorer;
