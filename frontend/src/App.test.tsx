import { render, screen } from "@testing-library/react";
import App from "./App";
import { DISCLAIMER } from "./constants";

describe("App", () => {
  it("renders the title", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: "GaitRAG" })).toBeInTheDocument();
  });

  it("always shows the disclaimer", () => {
    render(<App />);
    expect(screen.getByRole("note", { name: "Disclaimer" })).toHaveTextContent(DISCLAIMER);
  });
});
