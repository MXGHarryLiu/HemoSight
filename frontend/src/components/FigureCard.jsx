import React, { useEffect } from 'react';
import { Card } from 'react-bootstrap';
import embed from 'vega-embed';

export const FigureCard = ({ title, plot, index }) => {
  const plotID = `plot${index + 1}`;
  const plotTip = plot.usermeta.help;
  useEffect(() => {
    embed(`#${plotID}`, plot);
  }, [plot, plotID]);

  return (
    <Card className="mt-4">
      <Card.Header as="h5">{title}</Card.Header>
      <Card.Body>
        <Card.Text>{plotTip}</Card.Text>
        <div className="text-center" id={plotID} style={{ width: '100%' }}>
          <p>Please select a dataset first</p>
        </div>
      </Card.Body>
    </Card>
  );
}

export const fetchVegaPlot = async (name) => {
  const response = await fetch(`/api/vega?name=${name}`);
  const data = await response.json();
  return data;
}
