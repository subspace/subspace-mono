import { ApiPromise } from "@polkadot/api";
import { Logger } from "pino";
import { Block, Hash, SignedBlock } from "@polkadot/types/interfaces";
import { of } from "rxjs";

const block = {
  block: {
    header: {
      hash: "random hash",
    },
    extrinsics: [],
  },
};

export const apiMock = {
  rx: {
    rpc: {
      chain: {
        subscribeFinalizedHeads() {
          return of({
            hash: "random hash" as unknown as Hash,
          });
        },
      },
    },
  },
  rpc: {
    chain: {
      getBlock: jest.fn().mockResolvedValue(block as unknown as SignedBlock),
    },
  },
  query: {
    system: {
      events: {
        at() {
          return [];
        },
      },
    },
  },
  tx: {
    feeds: {
      put: jest.fn().mockReturnValue({
        signAndSend: jest
          .fn()
          .mockResolvedValue("random hash" as unknown as Hash),
      }),
    },
  },
} as unknown as ApiPromise;

export const loggerMock = {
  info: jest.fn(),
} as unknown as Logger;

export const fetchParaBlockMock = jest.fn().mockResolvedValue({} as Block);
